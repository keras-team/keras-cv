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
from tensorflow.keras import layers

from keras_cv import bounding_box
from keras_cv.ops import box_matcher
from keras_cv.ops import target_gather


class RetinaNetLabelEncoder(layers.Layer):
    """Transforms the raw labels into targets for training.

    This class has operations to generate targets for a batch of samples which
    is made up of the input images, bounding boxes for the objects present and
    their class ids.

    Args:
        bounding_box_format:  The format of bounding boxes of input dataset. Refer
            [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
            for more details on supported bounding box formats.
        anchor_generator: `keras_cv.layers.AnchorGenerator` instance to produce anchor
            boxes.  Boxes are then used to encode labels on a per-image basis.
        positive_threshold: the float threshold to set an anchor to positive match to gt box.
            values above it are positive matches.
        negative_threshold: the float threshold to set an anchor to negative match to gt box.
            values below it are negative matches.
        box_variance: The scaling factors used to scale the bounding box targets.
            Defaults to (0.1, 0.1, 0.2, 0.2).
        background_class: (Optional) The class ID used for the background class.
            Defaults to -1.
        ignore_class: (Optional) The class ID used for the ignore class. Defaults to -2.
    """

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
        self.box_variance = tf.convert_to_tensor(box_variance, dtype=self.dtype)
        self.background_class = background_class
        self.ignore_class = ignore_class
        self.matched_boxes_metric = tf.keras.metrics.BinaryAccuracy(
            name="percent_boxes_matched_with_anchor"
        )
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.box_matcher = box_matcher.ArgmaxBoxMatcher(
            thresholds=[negative_threshold, positive_threshold],
            match_values=[-1, -2, 1],
            force_match_for_each_col=False,
        )
        self.built = True

    def _encode_sample(self, gt_boxes, anchor_boxes):
        """Creates box and classification targets for a single sample"""
        cls_ids = gt_boxes["classes"]
        gt_boxes = gt_boxes["boxes"]
        cls_ids = tf.cast(cls_ids, dtype=self.dtype)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
            anchor_boxes, gt_boxes
        )
        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        box_target = bounding_box._encode_box_to_deltas(
            anchors=anchor_boxes,
            boxes=matched_gt_boxes,
            anchor_format=self.bounding_box_format,
            box_format="xywh",
            variance=self.box_variance,
        )
        matched_gt_cls_ids = target_gather._target_gather(gt_classes, matched_gt_idx)
        cls_target = tf.where(
            tf.not_equal(positive_mask, 1.0), self.background_class, matched_gt_cls_ids
        )
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), self.ignore_class, cls_target)

        box_target = tf.where(
            tf.math.is_nan(box_target),
            self.ignore_class,
            box_target,
        )
        cls_target = tf.where(
            tf.math.is_nan(cls_target),
            self.ignore_class,
            cls_target,
        )
        return {"boxes": box_target, "classes": cls_target}

        # In the case that a box in the corner of an image matches with an all -1 box
        # that is outside of the image, we should assign the box to the ignore class
        # There are rare cases where a -1 box can be matched, resulting in a NaN during
        # training.  The unit test passing all -1s to the label encoder ensures that we
        # properly handle this edge-case.

        #
        # n_boxes = tf.shape(gt_boxes)[0]
        # box_ids = tf.range(n_boxes, dtype=matched_gt_idx.dtype)
        # matched_ids = tf.expand_dims(matched_gt_idx, axis=1)
        # matches = box_ids == matched_ids
        # matches = tf.math.reduce_any(matches, axis=0)
        # self.matched_boxes_metric.update_state(
        #     tf.zeros((n_boxes,), dtype=tf.int32),
        #     tf.cast(matches, tf.int32),
        # )

    def call(self, images, bounding_boxes):
        """Creates box and classification targets for a batch

        Args:
          images: a batched [batch_size, H, W, C] image float `tf.Tensor`.
          boxes: a batched [batch_size, num_objects, 4] or ragged batch float ground truth boxes in `bounding_box_format`.
          classes: a batched [batch_size, num_objects] or ragged batch float ground truth classes.
        """
        if isinstance(images, tf.RaggedTensor):
            raise ValueError(
                "`RetinaNetLabelEncoder`'s `call()` method does not "
                "support RaggedTensor inputs for the `images` argument.  Received "
                f"`type(images)={type(images)}`."
            )
        bounding_boxes = bounding_box.to_dense(bounding_boxes)
        gt_boxes = tf.cast(bounding_boxes["boxes"], self.dtype)
        gt_classes = tf.cast(bounding_boxes["classes"], self.dtype)

        gt_boxes = bounding_box.convert_format(
            gt_boxes, source=self.bounding_box_format, target="xywh", images=images
        )
        anchor_boxes = self.anchor_generator(image_shape=tf.shape(images[0]))
        anchor_boxes = tf.concat(list(anchor_boxes.values()), axis=0)
        anchor_boxes = bounding_box.convert_format(
            anchor_boxes,
            source=self.anchor_generator.bounding_box_format,
            target=self.bounding_box_format,
            images=images[0],
        )

        result = self._encode_sample(gt_boxes, anchor_boxes)
        result = bounding_box.convert_format(
            result, source="xywh", target=self.bounding_box_format, images=images
        )
        encoded_box_targets = result["boxes"]
        class_targets = result["classes"]
        return encoded_box_targets, class_targets
