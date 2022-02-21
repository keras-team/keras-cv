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


def compute_ious_for_image(boxes1, boxes2):
    """computes a lookup table vector containing the ious for a given set boxes.

    The lookup vector is to be indexed by [`boxes1_index`,`boxes2_index`].

    Bounding boxes are expected to be in the corners format of
    `[bounding_box.LEFT, bounding_box.RIGHT, bounding_box.TOP, bounding_box.BOTTOM]`.
    For example, the bounding box with it's left side at 100, bounding_box.RIGHT side at
    200, bounding_box.TOP at 101, and bounding_box.BOTTOM at 201 would be represented
    as:
    > [100, 200, 101, 201]

    Args:
      boxes1: a list of bounding boxes in 'corners' format.
      boxes2: a list of bounding boxes in 'corners' format.

    Returns:
      iou_lookup_table: a vector containing the pairwise ious of boxes1 and
        boxes2.
    """
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
