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


def _compute_area(box):
    """Computes area for bounding boxes

    Args:
      box: [N, 4] or [batch_size, N, 4] float Tensor, either batched
        or unbatched boxes.
    Returns:
      a float Tensor of [N] or [batch_size, N]
    """
    y_min, x_min, y_max, x_max = tf.split(box[..., :4], num_or_size_splits=4, axis=-1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), axis=-1)


def _compute_intersection(boxes1, boxes2):
    """Computes intersection area between two sets of boxes.

    Args:
      boxes1: [N, 4] or [batch_size, N, 4] float Tensor boxes.
      boxes2: [M, 4] or [batch_size, M, 4] float Tensor boxes.
    Returns:
      a [N, M] or [batch_size, N, M] float Tensor.
    """
    y_min1, x_min1, y_max1, x_max1 = tf.split(
        boxes1[..., :4], num_or_size_splits=4, axis=-1
    )
    y_min2, x_min2, y_max2, x_max2 = tf.split(
        boxes2[..., :4], num_or_size_splits=4, axis=-1
    )
    boxes2_rank = len(boxes2.shape)
    perm = [1, 0] if boxes2_rank == 2 else [0, 2, 1]
    # [N, M] or [batch_size, N, M]
    intersect_ymax = tf.minimum(y_max1, tf.transpose(y_max2, perm))
    intersect_ymin = tf.maximum(y_min1, tf.transpose(y_min2, perm))
    intersect_xmax = tf.minimum(x_max1, tf.transpose(x_max2, perm))
    intersect_xmin = tf.maximum(x_min1, tf.transpose(x_min2, perm))

    intersect_height = intersect_ymax - intersect_ymin
    intersect_width = intersect_xmax - intersect_xmin
    zeros_t = tf.cast(0, intersect_height.dtype)
    intersect_height = tf.maximum(zeros_t, intersect_height)
    intersect_width = tf.maximum(zeros_t, intersect_width)

    return intersect_height * intersect_width


def compute_iou(
    boxes1,
    boxes2,
    bounding_box_format,
    use_masking=False,
    mask_val=-1,
):
    """Computes a lookup table vector containing the ious for a given set boxes.

    The lookup vector is to be indexed by [`boxes1_index`,`boxes2_index`] if boxes
    are unbatched and by [`batch`, `boxes1_index`,`boxes2_index`] if the boxes are
    batched.

    The users can pass `boxes1` and `boxes2` to be different ranks. For example:
    1) `boxes1`: [batch_size, M, 4], `boxes2`: [batch_size, N, 4] -> return [batch_size, M, N].
    2) `boxes1`: [batch_size, M, 4], `boxes2`: [N, 4] -> return [batch_size, M, N]
    3) `boxes1`: [M, 4], `boxes2`: [batch_size, N, 4] -> return [batch_size, M, N]
    4) `boxes1`: [M, 4], `boxes2`: [N, 4] -> return [M, N]

    Args:
      boxes1: a list of bounding boxes in 'corners' format. Can be batched or unbatched.
      boxes2: a list of bounding boxes in 'corners' format. Can be batched or unbatched.
      bounding_box_format: a case-insensitive string which is one of `"xyxy"`,
        `"rel_xyxy"`, `"xyWH"`, `"center_xyWH"`, `"yxyx"`, `"rel_yxyx"`.
        For detailed information on the supported format, see the
        [KerasCV bounding box documentation](https://keras.io/api/keras_cv/bounding_box/formats/).
    use_masking: whether masking will be applied. This will mask all `boxes1` or `boxes2` that
        have values less then 0 in all its 4 dimensions. Default to `False`.
    mask_val: int to mask those returned IOUs if the masking is True. Default to -1.

    Returns:
      iou_lookup_table: a vector containing the pairwise ious of boxes1 and
        boxes2.
    """

    boxes1_rank = len(boxes1.shape)
    boxes2_rank = len(boxes2.shape)

    if boxes1_rank not in [2, 3]:
        raise ValueError(
            "compute_iou() expects boxes1 to be batched, or "
            f"to be unbatched. Received len(boxes1.shape)={boxes1_rank}, "
            f"len(boxes2.shape)={boxes2_rank}. Expected either len(boxes1.shape)=2 AND "
            "or len(boxes1.shape)=3."
        )
    if boxes2_rank not in [2, 3]:
        raise ValueError(
            "compute_iou() expects boxes2 to be batched, or "
            f"to be unbatched. Received len(boxes1.shape)={boxes1_rank}, "
            f"len(boxes2.shape)={boxes2_rank}. Expected either len(boxes2.shape)=2 AND "
            "or len(boxes2.shape)=3."
        )

    target = bounding_box.preserve_rel(
        target_bounding_box_format="yxyx", bounding_box_format=bounding_box_format
    )

    boxes1 = bounding_box.convert_format(
        boxes1, source=bounding_box_format, target=target
    )

    boxes2 = bounding_box.convert_format(
        boxes2, source=bounding_box_format, target=target
    )

    intersect_area = _compute_intersection(boxes1, boxes2)
    boxes1_area = _compute_area(boxes1)
    boxes2_area = _compute_area(boxes2)
    boxes2_area_rank = len(boxes2_area.shape)
    boxes2_axis = 1 if (boxes2_area_rank == 2) else 0
    boxes1_area = tf.expand_dims(boxes1_area, axis=-1)
    boxes2_area = tf.expand_dims(boxes2_area, axis=boxes2_axis)
    union_area = boxes1_area + boxes2_area - intersect_area
    res = tf.math.divide_no_nan(intersect_area, union_area)

    if boxes1_rank == 2:
        perm = [1, 0]
    else:
        perm = [0, 2, 1]

    if not use_masking:
        return res

    mask_val_t = tf.cast(mask_val, res.dtype) * tf.ones_like(res)
    boxes1_mask = tf.less(tf.reduce_max(boxes1, axis=-1, keepdims=True), 0.0)
    boxes2_mask = tf.less(tf.reduce_max(boxes2, axis=-1, keepdims=True), 0.0)
    background_mask = tf.logical_or(boxes1_mask, tf.transpose(boxes2_mask, perm))
    iou_lookup_table = tf.where(background_mask, mask_val_t, res)
    return iou_lookup_table
