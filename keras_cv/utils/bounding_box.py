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
will contain 4 numbers based on 2 different formats:

1. LEFT, TOP, RIGHT, BOTTOM, where LEFT, TOP represent the top-left corner
   coordinates, and RIGHT, BOTTOM represent the bottom-right corner coordinates.
2. X, Y, WIDTH, HEIGHT, where X and Y are the coordinates for the center of the box.

Math wise:
LEFT = X - WIDTH / 2
TOP = Y - HEIGHT / 2
RIGHT = X + WIDTH / 2
BOTTOM = Y + HEIGHT / 2

X = (LEFT + RIGHT) / 2
Y = (TOP + BOTTOM) / 2
WIDTH = RIGHT - LEFT
HEIGHT = BOTTOM - TOP

Note that these two formats are both commonly used. Corners format are mostly used
for IOU computation, whereas XYWH are easy for bounding box generation with different
center and width/height ratio.
"""

import tensorflow as tf

# These are the indexes used in Tensors to represent each corresponding side.
LEFT, TOP, RIGHT, BOTTOM = 0, 1, 2, 3

# These are the indexes that you can use for bounding box in XYWH format.
X, Y, WIDTH, HEIGHT = 0, 1, 2, 3

# Regardless of format these constants are consistent.
# Class is held in the 5th index
CLASS = 4
# Confidence exists only on y_pred, and is in the 6th index.
CONFIDENCE = 5


def corners_to_xywh(bounding_boxes):
    """Converts bounding_boxes in corners format to XYWH format.

    Args:
        bounding_boxes: a Tensor which has at least 2D rank, with shape [..., 4]

    Returns:
        converted bounding_boxes with same shape, but in XYWH format.
    """
    left, top, right, bottom, rest = tf.split(bounding_boxes, [1, 1, 1, 1, -1], axis=-1)
    return tf.concat(
        [
            # We use ... here in case user has higher rank of inputs.
            (left + right) / 2.0,  # X
            (top + bottom) / 2.0,  # Y
            right - left,  # WIDTH
            bottom - top,  # HEIGHT
            rest,  # In case there is any more index after the BOTTOM.
        ],
        axis=-1,
    )


def xywh_to_corners(bounding_boxes):
    """Converts bounding_boxes in XYWH format to corners format.

    Args:
        bounding_boxes: a Tensor which has at least 2D rank, with shape [..., 4]

    Returns:
        converted bounding_boxes with same shape, but in corners format.
    """
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
