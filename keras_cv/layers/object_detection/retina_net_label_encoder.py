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
        self.built = True

    def _match_anchor_boxes(
        self, anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4
    ):
        """Matches ground truth boxes to anchor boxes based on IOU.
        1. Calculates the pairwise IOU for the M `anchor_boxes` and N `gt_boxes`
          to get a `(M, N)` shaped matrix.
        2. The ground truth box with the maximum IOU in each row is assigned to
          the anchor box provided the IOU is greater than `match_iou`.
        3. If the maximum IOU in a row is less than `ignore_iou`, the anchor
          box is assigned with the background class.
        4. The remaining anchor boxes that do not have any class assigned are
          ignored during training.
        Arguments:
          anchor_boxes: A float tensor with the shape `(total_anchors, 4)`
            representing all the anchor boxes for a given input image shape,
            where each anchor box is of the format `[x, y, width, height]`.
          gt_boxes: A float tensor with shape `(num_objects, 4)` representing
            the ground truth boxes, where each box is of the format
            `[x, y, width, height]`.
          match_iou: A float value representing the minimum IOU threshold for
            determining if a ground truth box can be assigned to an anchor box.
          ignore_iou: A float value representing the IOU threshold under which
            an anchor box is assigned to the background class.
        Returns:
          matched_gt_idx: Index of the matched object
          positive_mask: A mask for anchor boxes that have been assigned ground
            truth boxes.
          ignore_mask: A mask for anchor boxes that need to by ignored during
            training
        """
        iou_matrix = bounding_box.compute_iou(
            anchor_boxes, gt_boxes, bounding_box_format="xywh"
        )
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=self.dtype),
            tf.cast(ignore_mask, dtype=self.dtype),
        )

    def _encode_sample(self, gt_boxes, anchor_boxes):
        """Creates box and classification targets for a single sample"""
        cls_ids = gt_boxes[:, 4]
        gt_boxes = gt_boxes[:, :4]
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
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        cls_target = tf.where(
            tf.not_equal(positive_mask, 1.0), self.background_class, matched_gt_cls_ids
        )
        cls_target = tf.where(tf.equal(ignore_mask, 1.0), self.ignore_class, cls_target)
        cls_target = tf.expand_dims(cls_target, axis=-1)
        label = tf.concat([box_target, cls_target], axis=-1)

        # In the case that a box in the corner of an image matches with an all -1 box
        # that is outside of the image, we should assign the box to the ignore class
        # There are rare cases where a -1 box can be matched, resulting in a NaN during
        # training.  The unit test passing all -1s to the label encoder ensures that we
        # properly handle this edge-case.
        label = tf.where(
            tf.expand_dims(tf.math.reduce_any(tf.math.is_nan(label), axis=-1), axis=-1),
            self.ignore_class,
            label,
        )

        n_boxes = tf.shape(gt_boxes)[0]
        box_ids = tf.range(n_boxes, dtype=matched_gt_idx.dtype)
        matched_ids = tf.expand_dims(matched_gt_idx, axis=1)
        matches = box_ids == matched_ids
        matches = tf.math.reduce_any(matches, axis=0)
        self.matched_boxes_metric.update_state(
            tf.zeros((n_boxes,), dtype=tf.int32),
            tf.cast(matches, tf.int32),
        )
        return label

    def call(self, images, target_boxes):
        """Creates box and classification targets for a batch"""
        if isinstance(images, tf.RaggedTensor):
            raise ValueError(
                "`RetinaNetLabelEncoder`'s `call()` method does not "
                "support RaggedTensor inputs for the `images` argument.  Received "
                f"`type(images)={type(images)}`."
            )

        target_boxes = bounding_box.convert_format(
            target_boxes, source=self.bounding_box_format, target="xywh", images=images
        )
        anchor_boxes = self.anchor_generator(image_shape=tf.shape(images[0]))
        anchor_boxes = tf.concat(list(anchor_boxes.values()), axis=0)
        anchor_boxes = bounding_box.convert_format(
            anchor_boxes,
            source=self.anchor_generator.bounding_box_format,
            target=self.bounding_box_format,
            images=images[0],
        )

        if isinstance(target_boxes, tf.RaggedTensor):
            target_boxes = target_boxes.to_tensor(
                default_value=-1, shape=(None, None, 5)
            )

        result = tf.map_fn(
            elems=(target_boxes),
            fn=lambda box_set: self._encode_sample(box_set, anchor_boxes),
        )
        return bounding_box.convert_format(
            result, source="xywh", target=self.bounding_box_format, images=images
        )
