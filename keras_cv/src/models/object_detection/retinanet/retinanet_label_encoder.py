# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

from keras_cv.src import bounding_box
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.layers.object_detection import box_matcher
from keras_cv.src.utils import target_gather


@keras_cv_export("keras_cv.models.retinanet.LabelEncoder")
class RetinaNetLabelEncoder(keras.layers.Layer):
    """Transforms the raw labels into targets for training.

    This class has operations to generate targets for a batch of samples which
    is made up of the input images, bounding boxes for the objects present and
    their class ids. Targets are always represented in `center_yxwh` format.
    This done for numerical reasons, to ensure numerical consistency when
    training in any format.

    Args:
        bounding_box_format: The format of bounding boxes of input dataset.
            Refer [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/) for more
            details on supported bounding box formats.
        anchor_generator: `keras_cv.layers.AnchorGenerator` instance to produce
            anchor boxes. Boxes are then used to encode labels on a per-image
            basis.
        positive_threshold: the float threshold to set an anchor to positive
            match to gt box. Values above it are positive matches.
        negative_threshold: the float threshold to set an anchor to negative
            match to gt box. Values below it are negative matches.
        box_variance: The scaling factors used to scale the bounding box
            targets, defaults to (0.1, 0.1, 0.2, 0.2).
        background_class: (Optional) The class ID used for the background class,
            defaults to -1.
        ignore_class: (Optional) The class ID used for the ignore class,
            defaults to -2.
    """  # noqa: E501

    def __init__(
        self,
        bounding_box_format,
        anchor_generator,
        positive_threshold=0.5,
        negative_threshold=0.4,
        box_variance=(0.1, 0.1, 0.2, 0.2),
        background_class=-1.0,
        ignore_class=-2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format
        self.anchor_generator = anchor_generator
        self.box_variance = ops.array(box_variance, "float32")
        self.background_class = background_class
        self.ignore_class = ignore_class
        self.matched_boxes_metric = MatchedBoxesMetric(
            name="percent_boxes_matched_with_anchor"
        )
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.box_matcher = box_matcher.BoxMatcher(
            thresholds=[negative_threshold, positive_threshold],
            match_values=[-1, -2, 1],
            force_match_for_each_col=False,
        )
        self.box_variance_tuple = box_variance
        self.built = True

    def _encode_sample(self, box_labels, anchor_boxes, image_shape):
        """Creates box and classification targets for a batched sample
        Matches ground truth boxes to anchor boxes based on IOU.
        1. Calculates the pairwise IOU for the M `anchor_boxes` and N `gt_boxes`
          to get a `(M, N)` shaped matrix.
        2. The ground truth box with the maximum IOU in each row is assigned to
          the anchor box provided the IOU is greater than `match_iou`.
        3. If the maximum IOU in a row is less than `ignore_iou`, the anchor
          box is assigned with the background class.
        4. The remaining anchor boxes that do not have any class assigned are
          ignored during training.
        Args:
          gt_boxes: A float tensor with shape `(num_objects, 4)` representing
            the ground truth boxes, where each box is of the format
            `[x, y, width, height]`.
          gt_classes: A float Tensor with shape `(num_objects, 1)` representing
            the ground truth classes.
          anchor_boxes: A float tensor with the shape `(total_anchors, 4)`
            representing all the anchor boxes for a given input image shape,
            where each anchor box is of the format `[x, y, width, height]`.
        Returns:
          matched_gt_idx: Index of the matched object
          positive_mask: A mask for anchor boxes that have been assigned ground
            truth boxes.
          ignore_mask: A mask for anchor boxes that need to by ignored during
            training
        """
        gt_boxes = box_labels["boxes"]
        gt_classes = box_labels["classes"]
        iou_matrix = bounding_box.compute_iou(
            anchor_boxes,
            gt_boxes,
            bounding_box_format=self.bounding_box_format,
            image_shape=image_shape,
        )
        matched_gt_idx, matched_vals = self.box_matcher(iou_matrix)
        matched_vals = ops.expand_dims(matched_vals, axis=-1)
        positive_mask = ops.cast(ops.equal(matched_vals, 1), self.dtype)
        ignore_mask = ops.cast(ops.equal(matched_vals, -2), self.dtype)
        matched_gt_boxes = target_gather._target_gather(
            gt_boxes, matched_gt_idx
        )
        matched_gt_boxes = ops.reshape(
            matched_gt_boxes, (-1, ops.shape(matched_gt_boxes)[1], 4)
        )

        box_target = bounding_box._encode_box_to_deltas(
            anchors=anchor_boxes,
            boxes=matched_gt_boxes,
            anchor_format=self.bounding_box_format,
            box_format=self.bounding_box_format,
            variance=self.box_variance,
            image_shape=image_shape,
        )
        matched_gt_cls_ids = target_gather._target_gather(
            gt_classes, matched_gt_idx
        )
        cls_target = ops.where(
            ops.not_equal(positive_mask, 1.0),
            self.background_class,
            matched_gt_cls_ids,
        )
        cls_target = ops.where(
            ops.equal(ignore_mask, 1.0), self.ignore_class, cls_target
        )
        label = ops.concatenate(
            [box_target, ops.cast(cls_target, box_target.dtype)], axis=-1
        )

        # In the case that a box in the corner of an image matches with an all
        # -1 box that is outside the image, we should assign the box to the
        # ignore class. There are rare cases where a -1 box can be matched,
        # resulting in a NaN during training. The unit test passing all -1s to
        # the label encoder ensures that we properly handle this edge-case.
        label = ops.where(
            ops.expand_dims(ops.any(ops.isnan(label), axis=-1), axis=-1),
            self.ignore_class,
            label,
        )

        result = {"boxes": label[:, :, :4], "classes": label[:, :, 4]}

        box_shape = ops.shape(gt_boxes)
        batch_size = box_shape[0]
        n_boxes = box_shape[1]
        box_ids = ops.arange(n_boxes, dtype=matched_gt_idx.dtype)
        matched_ids = ops.expand_dims(matched_gt_idx, axis=-1)
        matches = box_ids == matched_ids
        matches = ops.any(matches, axis=1)
        self.matched_boxes_metric.update_state(
            ops.zeros(
                (
                    batch_size,
                    n_boxes,
                ),
                dtype="int32",
            ),
            ops.cast(matches, "int32"),
        )
        return result

    def call(self, images, box_labels):
        """Creates box and classification targets for a batch

        Args:
          images: a batched [batch_size, H, W, C] image float `tf.Tensor`.
          box_labels: a batched KerasCV style bounding box dictionary containing
            bounding boxes and class labels. Should be in `bounding_box_format`.
        """
        if isinstance(images, tf.RaggedTensor):
            raise ValueError(
                "`RetinaNetLabelEncoder`'s `call()` method does not "
                "support RaggedTensor inputs for the `images` argument. "
                f"Received `type(images)={type(images)}`."
            )

        image_shape = ops.shape(images)
        image_shape = (image_shape[1], image_shape[2], image_shape[3])
        box_labels = bounding_box.to_dense(box_labels)
        if len(box_labels["classes"].shape) == 2:
            box_labels["classes"] = ops.expand_dims(
                box_labels["classes"], axis=-1
            )
        anchor_boxes = self.anchor_generator(image_shape=image_shape)
        anchor_boxes = ops.concatenate(list(anchor_boxes.values()), axis=0)
        anchor_boxes = bounding_box.convert_format(
            anchor_boxes,
            source=self.anchor_generator.bounding_box_format,
            target=self.bounding_box_format,
            image_shape=image_shape,
        )

        result = self._encode_sample(box_labels, anchor_boxes, image_shape)
        encoded_box_targets = result["boxes"]
        encoded_box_targets = ops.reshape(
            encoded_box_targets, (-1, ops.shape(encoded_box_targets)[1], 4)
        )
        class_targets = result["classes"]
        return encoded_box_targets, class_targets

    def get_config(self):
        config = {
            "bounding_box_format": self.bounding_box_format,
            "anchor_generator": self.anchor_generator,
            "positive_threshold": self.positive_threshold,
            "negative_threshold": self.negative_threshold,
            "box_variance": self.box_variance_tuple,
            "background_class": self.background_class,
            "ignore_class": self.ignore_class,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if "anchor_generator" in config and isinstance(
            config["anchor_generator"], dict
        ):
            config["anchor_generator"] = keras.layers.deserialize(
                config["anchor_generator"]
            )

        return super().from_config(config)


class MatchedBoxesMetric(keras.metrics.BinaryAccuracy):
    # Prevent `load_weights` from accessing metric
    def load_own_variables(self, store):
        return
