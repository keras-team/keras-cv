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
import math

from keras_cv import bounding_box
from keras_cv.backend import keras
from keras_cv.backend import ops


def _compute_area(box):
    """Computes area for bounding boxes

    Args:
      box: [N, 4] or [batch_size, N, 4] float Tensor, either batched
        or unbatched boxes.
    Returns:
      a float Tensor of [N] or [batch_size, N]
    """
    y_min, x_min, y_max, x_max = ops.split(box[..., :4], 4, axis=-1)
    return ops.squeeze((y_max - y_min) * (x_max - x_min), axis=-1)


def _compute_intersection(boxes1, boxes2):
    """Computes intersection area between two sets of boxes.

    Args:
      boxes1: [N, 4] or [batch_size, N, 4] float Tensor boxes.
      boxes2: [M, 4] or [batch_size, M, 4] float Tensor boxes.
    Returns:
      a [N, M] or [batch_size, N, M] float Tensor.
    """
    y_min1, x_min1, y_max1, x_max1 = ops.split(boxes1[..., :4], 4, axis=-1)
    y_min2, x_min2, y_max2, x_max2 = ops.split(boxes2[..., :4], 4, axis=-1)
    boxes2_rank = len(boxes2.shape)
    perm = [1, 0] if boxes2_rank == 2 else [0, 2, 1]
    # [N, M] or [batch_size, N, M]
    intersect_ymax = ops.minimum(y_max1, ops.transpose(y_max2, perm))
    intersect_ymin = ops.maximum(y_min1, ops.transpose(y_min2, perm))
    intersect_xmax = ops.minimum(x_max1, ops.transpose(x_max2, perm))
    intersect_xmin = ops.maximum(x_min1, ops.transpose(x_min2, perm))

    intersect_height = intersect_ymax - intersect_ymin
    intersect_width = intersect_xmax - intersect_xmin
    zeros_t = ops.cast(0, intersect_height.dtype)
    intersect_height = ops.maximum(zeros_t, intersect_height)
    intersect_width = ops.maximum(zeros_t, intersect_width)

    return intersect_height * intersect_width


def compute_iou(
    boxes1,
    boxes2,
    bounding_box_format,
    use_masking=False,
    mask_val=-1,
    images=None,
    image_shape=None,
):
    """Computes a lookup table vector containing the ious for a given set boxes.

    The lookup vector is to be indexed by [`boxes1_index`,`boxes2_index`] if
    boxes are unbatched and by [`batch`, `boxes1_index`,`boxes2_index`] if the
    boxes are batched.

    The users can pass `boxes1` and `boxes2` to be different ranks. For example:
    1) `boxes1`: [batch_size, M, 4], `boxes2`: [batch_size, N, 4] -> return
        [batch_size, M, N].
    2) `boxes1`: [batch_size, M, 4], `boxes2`: [N, 4] -> return
        [batch_size, M, N]
    3) `boxes1`: [M, 4], `boxes2`: [batch_size, N, 4] -> return
        [batch_size, M, N]
    4) `boxes1`: [M, 4], `boxes2`: [N, 4] -> return [M, N]

    Args:
      boxes1: a list of bounding boxes in 'corners' format. Can be batched or
        unbatched.
      boxes2: a list of bounding boxes in 'corners' format. Can be batched or
        unbatched.
      bounding_box_format: a case-insensitive string which is one of `"xyxy"`,
        `"rel_xyxy"`, `"xyWH"`, `"center_xyWH"`, `"yxyx"`, `"rel_yxyx"`.
        For detailed information on the supported format, see the
        [KerasCV bounding box documentation](https://keras.io/api/keras_cv/bounding_box/formats/).
      use_masking: whether masking will be applied. This will mask all `boxes1`
        or `boxes2` that have values less than 0 in all its 4 dimensions.
        Default to `False`.
      mask_val: int to mask those returned IOUs if the masking is True, defaults
        to -1.

    Returns:
      iou_lookup_table: a vector containing the pairwise ious of boxes1 and
        boxes2.
    """  # noqa: E501

    boxes1_rank = len(boxes1.shape)
    boxes2_rank = len(boxes2.shape)

    if boxes1_rank not in [2, 3]:
        raise ValueError(
            "compute_iou() expects boxes1 to be batched, or to be unbatched. "
            f"Received len(boxes1.shape)={boxes1_rank}, "
            f"len(boxes2.shape)={boxes2_rank}. Expected either "
            "len(boxes1.shape)=2 AND or len(boxes1.shape)=3."
        )
    if boxes2_rank not in [2, 3]:
        raise ValueError(
            "compute_iou() expects boxes2 to be batched, or to be unbatched. "
            f"Received len(boxes1.shape)={boxes1_rank}, "
            f"len(boxes2.shape)={boxes2_rank}. Expected either "
            "len(boxes2.shape)=2 AND or len(boxes2.shape)=3."
        )

    target_format = "yxyx"
    if bounding_box.is_relative(bounding_box_format):
        target_format = bounding_box.as_relative(target_format)

    boxes1 = bounding_box.convert_format(
        boxes1,
        source=bounding_box_format,
        target=target_format,
        images=images,
        image_shape=image_shape,
    )

    boxes2 = bounding_box.convert_format(
        boxes2,
        source=bounding_box_format,
        target=target_format,
        images=images,
        image_shape=image_shape,
    )

    intersect_area = _compute_intersection(boxes1, boxes2)
    boxes1_area = _compute_area(boxes1)
    boxes2_area = _compute_area(boxes2)
    boxes2_area_rank = len(boxes2_area.shape)
    boxes2_axis = 1 if (boxes2_area_rank == 2) else 0
    boxes1_area = ops.expand_dims(boxes1_area, axis=-1)
    boxes2_area = ops.expand_dims(boxes2_area, axis=boxes2_axis)
    union_area = boxes1_area + boxes2_area - intersect_area
    res = ops.divide(intersect_area, union_area + keras.backend.epsilon())

    if boxes1_rank == 2:
        perm = [1, 0]
    else:
        perm = [0, 2, 1]

    if not use_masking:
        return res

    mask_val_t = ops.cast(mask_val, res.dtype) * ops.ones_like(res)
    boxes1_mask = ops.less(ops.max(boxes1, axis=-1, keepdims=True), 0.0)
    boxes2_mask = ops.less(ops.max(boxes2, axis=-1, keepdims=True), 0.0)
    background_mask = ops.logical_or(
        boxes1_mask, ops.transpose(boxes2_mask, perm)
    )
    iou_lookup_table = ops.where(background_mask, mask_val_t, res)
    return iou_lookup_table


def compute_ciou(box1, box2, bounding_box_format, eps=1e-7):
    """
    Computes the Complete IoU (CIoU) between two bounding boxes or between
    two batches of bounding boxes.

    CIoU loss is an extension of GIoU loss, which further improves the IoU
    optimization for object detection. CIoU loss not only penalizes the
    bounding box coordinates but also considers the aspect ratio and center
    distance of the boxes. The length of the last dimension should be 4 to
    represent the bounding boxes.

    Args:
        box1 (tensor): tensor representing the first bounding box with
            shape (..., 4).
        box2 (tensor): tensor representing the second bounding box with
            shape (..., 4).
        bounding_box_format: a case-insensitive string (for example, "xyxy").
            Each bounding box is defined by these 4 values. For detailed
            information on the supported formats, see the [KerasCV bounding box
            documentation](https://keras.io/api/keras_cv/bounding_box/formats/).
        eps (float, optional): A small value to avoid division by zero. Default
            is 1e-7.

    Returns:
        tensor: The CIoU distance between the two bounding boxes.
    """
    target_format = "xyxy"
    if bounding_box.is_relative(bounding_box_format):
        target_format = bounding_box.as_relative(target_format)

    box1 = bounding_box.convert_format(
        box1, source=bounding_box_format, target=target_format
    )

    box2 = bounding_box.convert_format(
        box2, source=bounding_box_format, target=target_format
    )
    b1_x1, b1_y1, b1_x2, b1_y2 = ops.split(box1, 4, axis=-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = ops.split(box2, 4, axis=-1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = ops.maximum(
        ops.minimum(b1_x2, b2_x2) - ops.maximum(b1_x1, b2_x1), 0
    ) * ops.maximum(ops.minimum(b1_y2, b2_y2) - ops.maximum(b1_y1, b2_y1), 0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    cw = ops.maximum(b1_x2, b2_x2) - ops.minimum(
        b1_x1, b2_x1
    )  # convex (smallest enclosing box) width
    ch = ops.maximum(b1_y2, b2_y2) - ops.minimum(b1_y1, b2_y1)  # convex height
    c2 = cw**2 + ch**2 + eps  # convex diagonal squared
    rho2 = (
        (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
        + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
    ) / 4  # center dist ** 2
    v = ops.power(
        (4 / math.pi**2) * (ops.arctan(w2 / h2) - ops.arctan(w1 / h1)), 2
    )
    alpha = v / (v - iou + (1 + eps))

    return iou - (rho2 / c2 + v * alpha)
