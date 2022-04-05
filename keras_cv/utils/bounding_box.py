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

"""Shared utility functions for working with bounding boxes.

Usually bounding boxes is a 2D Tensor with shape [batch, 4]. The second dimension
will contain 4 numbers based on 2 different formats.  In KerasCV, we will use the
`corners` format, which is [LEFT, TOP, RIGHT, BOTTOM].

In this file, provide utility functions for manipulating bounding boxes and converting
their formats.
"""

import tensorflow as tf

# These are the indexes used in Tensors to represent each corresponding side.
LEFT, TOP, RIGHT, BOTTOM = 0, 1, 2, 3

# Regardless of format these constants are consistent.
# Class is held in the 5th index
CLASS = 4
# Confidence exists only on y_pred, and is in the 6th index.
CONFIDENCE = 5


def convert_to_corners(bounding_boxes, format):
    """Converts bounding_boxes to corners format.

    Converts bounding boxes from the provided format to corners format, which is:
    `[left, top, right, bottom]`.

    args:
        format:  one of "coco" or "yolo".  The formats are as follows-
            coco=[x_min, y_min, width, height]
            yolo=[x_center, y_center, width, height]
    """
    if format == "coco":
        return _coco_to_corners(bounding_boxes)
    elif format == "yolo":
        return _yolo_to_corners(bounding_boxes)
    else:
        raise ValueError(
            "Unsupported format passed to convert_to_corners().  "
            f"Want one 'coco' or 'yolo', got format=={format}"
        )


def _yolo_to_corners(bounding_boxes):
    x, y, width, height, rest = tf.split(bounding_boxes, [1, 1, 1, 1, -1], axis=-1)
    return tf.concat(
        [
            x - width / 2.0,
            y - height / 2.0,
            x + width / 2.0,
            y + height / 2.0,
            rest,  # In case there is any more index after the HEIGHT.
        ],
        axis=-1,
    )


def _coco_to_corners(bounding_boxes):
    x, y, width, height, rest = tf.split(bounding_boxes, [1, 1, 1, 1, -1], axis=-1)
    return tf.concat(
        [
            x,
            y,
            x + width,
            y + height,
            rest,  # In case there is any more index after the HEIGHT.
        ],
        axis=-1,
    )


def pad_bounding_box_batch_to_shape(bounding_boxes, target_shape, padding_values=-1):
    """Pads a list of bounding boxes with -1s.

    Boxes represented by all -1s are ignored by COCO metrics.

    Sample usage:
    bounding_box = [[1, 2, 3, 4], [5, 6, 7, 8]]   # 2 bounding_boxes with with xywh or
        corners format.
    target_shape = [3, 4]   # Add 1 more dummy bounding_box
    result = pad_bounding_box_batch_to_shape(bounding_box, target_shape)
    # result == [[1, 2, 3, 4], [5, 6, 7, 8], [-1, -1, -1, -1]]

    target_shape = [2, 5]   # Add 1 more index after the current 4 coordinates.
    result = pad_bounding_box_batch_to_shape(bounding_box, target_shape)
    # result == [[1, 2, 3, 4, -1], [5, 6, 7, 8, -1]]

    Args:
        bounding_boxes: tf.Tensor of bounding boxes in any format.
        target_shape: Target shape to pad bounding box to. This should have the same
            rank as the bbounding_boxs. Note that if the target_shape contains any
            dimension that is smaller than the bounding box shape, then no value will be
            padded.
        padding_values: value to pad, defaults to -1 to mask out in coco metrics.
    Returns:
        bounding_boxes padded to target shape.

    Raises:
        ValueError, when target shape has smaller rank or dimension value when
            comparing with shape of bounding boxes.
    """
    bounding_box_shape = tf.shape(bounding_boxes)
    if len(bounding_box_shape) != len(target_shape):
        raise ValueError(
            "Target shape should have same rank as the bounding box. "
            f"Got bounding_box shape = {bounding_box_shape}, "
            f"target_shape = {target_shape}"
        )
    for dim in range(len(target_shape)):
        if bounding_box_shape[dim] > target_shape[dim]:
            raise ValueError(
                "Target shape should be larger than bounding box shape "
                "in all dimensions. "
                f"Got bounding_box shape = {bounding_box_shape}, "
                f"target_shape = {target_shape}"
            )
    paddings = [
        [0, target_shape[dim] - bounding_box_shape[dim]]
        for dim in range(len(target_shape))
    ]
    return tf.pad(
        bounding_boxes, paddings, mode="CONSTANT", constant_values=padding_values
    )


def mask_to_bounding_boxes(masks, batch_dim=0):
    """Computes bouding boxes from masks.

    Args:
        masks: tf.Tensor of binary masks.
        batch_dim (optional): Perform bouding box extraction on given dimension.
    Returns:
        Bounding_boxes extracted from binary masks, if no bounding box found,
        dummy bounding box will be returned i.e [-1, -1, -1, -1].

    Raises:
        ValueError, when `masks` ndim is not in compatible range.
    """

    def _get_bounding_box(mask):
        def _get_positive_pixel_coordinate(group):
            positive_indices = tf.cast(tf.where(group), tf.int32)
            is_empty = tf.equal(tf.size(positive_indices), 0)
            if is_empty:
                _dummy = tf.constant([-1], tf.int32)
                return _dummy, _dummy
            return positive_indices[0], positive_indices[-1]

        # convert rows to bool by nonzero pixel.
        rows = tf.math.count_nonzero(mask, axis=1, keepdims=None, dtype=tf.bool)
        # convert columns to bool by nonzero pixel.
        columns = tf.math.count_nonzero(mask, axis=0, keepdims=None, dtype=tf.bool)

        # get first and last pos occurence i.e TOP BOTTOM pixel.
        top, bottom = _get_positive_pixel_coordinate(rows)
        # get first and last pos occurence i.e LEFT RIGHT pixel.
        left, right = _get_positive_pixel_coordinate(columns)
        return tf.concat([top, left, bottom, right], axis=-1, dtype=tf.int32)

    _ndim = masks.ndim
    if _ndim < 2:
        raise ValueError(f"Masks should be at least 2 dimensional but {_ndim} passed.")
    elif _ndim == 2:
        return _get_bounding_box(masks)

    # replace first element with `batch_dim`
    _perm = list(range(_ndim))
    _perm = [_perm.pop(batch_dim)] + _perm

    # unroll `batch_dim` on first dimension.
    masks = tf.transpose(masks, _perm)
    return tf.map_fn(_get_bounding_box, masks, dtype=tf.int32)
