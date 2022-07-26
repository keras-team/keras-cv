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
"""Contains functions to compute ious of bounding boxes."""
import tensorflow as tf

from keras_cv import bounding_box


def compute_iou(boxes1, boxes2, bounding_box_format):
    """Computes a lookup table vector containing the ious for a given set boxes.

    The lookup vector is to be indexed by [`boxes1_index`,`boxes2_index`] if boxes
    are unbatched and by [`batch`, `boxes1_index`,`boxes2_index`] if the boxes are
    batched.

    Args:
      boxes1: a list of bounding boxes in 'corners' format. Can be batched or unbatched.
      boxes2: a list of bounding boxes in 'corners' format. This should match the rank and
        shape of boxes1.
      bounding_box_format: a case-sensitive string which is one of `"xyxy"`,
        `"rel_xyxy"`, `"xyWH"`, `"center_xyWH"`, `"yxyx"`, `"rel_yxyx"`.
        For detailed information on the supported format, see the
        [KerasCV bounding box documentation](https://keras.io/api/keras_cv/bounding_box/formats/).

    Returns:
      iou_lookup_table: a vector containing the pairwise ious of boxes1 and
        boxes2.
    """

    boxes1_rank = len(boxes1.shape)
    boxes2_rank = len(boxes2.shape)

    if (
        boxes1_rank != boxes2_rank
        or boxes1_rank not in [2, 3]
        or boxes2_rank not in [2, 3]
    ):
        raise ValueError(
            "compute_iou() expects both boxes to be batched, or both "
            f"boxes to be unbatched.  Received len(boxes1.shape)={boxes1_rank}, "
            f"len(boxes2.shape)={boxes2_rank}.  Expected either len(boxes1.shape)=2 AND "
            "len(boxes2.shape)=2, or len(boxes1.shape)=3 AND len(boxes2.shape)=3."
        )

    if bounding_box_format.startswith("rel"):
        target = "rel_yxyx"
    else:
        target = "yxyx"

    boxes1 = bounding_box.convert_format(
        boxes1, source=bounding_box_format, target=target
    )

    boxes2 = bounding_box.convert_format(
        boxes2, source=bounding_box_format, target=target
    )

    def compute_iou_for_batch(boxes):
        boxes1, boxes2 = boxes
        zero = tf.convert_to_tensor(0.0, boxes1.dtype)
        boxes1_ymin, boxes1_xmin, boxes1_ymax, boxes1_xmax = tf.unstack(
            boxes1[..., :4, None], 4, axis=-2
        )
        boxes2_ymin, boxes2_xmin, boxes2_ymax, boxes2_xmax = tf.unstack(
            boxes2[None, ..., :4], 4, axis=-1
        )
        boxes1_width = tf.maximum(zero, boxes1_xmax - boxes1_xmin)
        boxes1_height = tf.maximum(zero, boxes1_ymax - boxes1_ymin)
        boxes2_width = tf.maximum(zero, boxes2_xmax - boxes2_xmin)
        boxes2_height = tf.maximum(zero, boxes2_ymax - boxes2_ymin)
        boxes1_area = boxes1_width * boxes1_height
        boxes2_area = boxes2_width * boxes2_height
        intersect_ymin = tf.maximum(boxes1_ymin, boxes2_ymin)
        intersect_xmin = tf.maximum(boxes1_xmin, boxes2_xmin)
        intersect_ymax = tf.minimum(boxes1_ymax, boxes2_ymax)
        intersect_xmax = tf.minimum(boxes1_xmax, boxes2_xmax)
        intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
        intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
        intersect_area = intersect_width * intersect_height

        union_area = boxes1_area + boxes2_area - intersect_area
        return tf.math.divide_no_nan(intersect_area, union_area)

    if boxes1_rank == 2:
        return compute_iou_for_batch((boxes1, boxes2))
    else:
        return tf.map_fn(
            compute_iou_for_batch, elems=(boxes1, boxes2), dtype=boxes1.dtype
        )
