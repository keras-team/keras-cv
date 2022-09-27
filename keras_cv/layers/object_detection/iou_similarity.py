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

from keras_cv import bounding_box


class IouSimilarity(tf.keras.layers.Layer):
    """Computes a lookup table matrix containing the ious for a given pair of boxes.

    The lookup vector is to be indexed by [`boxes1_index`,`boxes2_index`] if boxes
    are unbatched and by [`batch`, `boxes1_index`,`boxes2_index`] if the boxes are
    batched.

    Args:
      boxes1: a list of bounding boxes in 'corners' format. Can be batched or unbatched.
      boxes2: a list of bounding boxes in 'corners' format. This should match the rank and
        shape of boxes1.
      bounding_box_format: a case-insensitive string.
        For detailed information on the supported format, see the
        [KerasCV bounding box documentation](https://keras.io/api/keras_cv/bounding_box/formats/).

    Returns:
      iou_lookup_table: a vector containing the pairwise ious of boxes1 and
        boxes2.
    """

    def __init__(
        self,
        box1_format,
        box2_format,
        box1_mask=False,
        box2_mask=False,
        mask_val=-1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.box1_format = box1_format
        self.box2_format = box2_format
        self.mask_val = mask_val
        self.box1_mask = box1_mask
        self.box2_mask = box2_mask
        self.built = True

    def call(self, boxes1, boxes2, boxes1_mask=None, boxes2_mask=None):
        boxes1 = bounding_box.convert_format(
            boxes1, source=self.box1_format, target="xyxy"
        )
        boxes2 = bounding_box.convert_format(
            boxes2, source=self.box2_format, target="xyxy"
        )
        result = bounding_box.compute_iou(boxes1, boxes2, bounding_box_format="xyxy")
        if not self.box1_mask and not self.box2_mask:
            return result
        background_mask = None
        boxes2_rank = len(boxes2.shape)
        mask_val_t = tf.cast(self.mask_val, result.dtype) * tf.ones_like(result)
        perm = [1, 0] if boxes2_rank == 2 else [0, 2, 1]
        if self.box1_mask and self.box2_mask:
            boxes1_mask = tf.less(tf.reduce_max(boxes1, axis=-1, keepdims=True), 0.0)
            boxes2_mask = tf.less(tf.reduce_max(boxes2, axis=-1, keepdims=True), 0.0)
            background_mask = tf.logical_or(
                boxes1_mask, tf.transpose(boxes2_mask, perm)
            )
        elif self.box1_mask:
            boxes1_mask = tf.less(tf.reduce_max(boxes1, axis=-1, keepdims=True), 0.0)
            background_mask = boxes1_mask
        else:
            boxes2_mask = tf.less(tf.reduce_max(boxes2, axis=-1, keepdims=True), 0.0)
            background_mask = tf.logical_or(
                tf.zeros(tf.shape(boxes2)[:-1], dtype=tf.bool),
                tf.transpose(boxes2_mask, perm),
            )
        iou_lookup_table = tf.where(background_mask, mask_val_t, result)
        return iou_lookup_table

    def get_config(self):
        config = {
            "box1_format": self.box1_format,
            "box2_format": self.box2_format,
            "box1_mask": self.box1_mask,
            "box2_mask": self.box2_mask,
            "mask_val": self.mask_val,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
