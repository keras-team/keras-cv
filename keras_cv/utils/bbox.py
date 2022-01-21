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

"""Shared utility functions for working with bounding boxes."""

import tensorflow as tf

# These are the dimensions used in Tensors to represent each corresponding side.
LEFT, TOP, RIGHT, BOTTOM = 0, 1, 2, 3

# These are the dimensions that you can use for bboxes in corners format.
X, Y, WIDTH, HEIGHT = 0, 1, 2, 3

# Regardless of format these constants are consistent.
# Class is held in the 4th index
CLASS = 4
# Confidence exists only on y_pred, and is in the 5th index.
CONFIDENCE = 5


def convert_corners_to_xywh(bboxes):
    """Converts bboxes in corners format to xywh format."""
    return tf.concat(
        [
            (bboxes[..., :2] + bboxes[..., 2:4]) / 2.0,
            bboxes[..., 2:4] - bboxes[..., :2],
            bboxes[..., 4:],
        ],
        axis=-1,
    )


def xywh_to_corners(bboxes):
    """Converts bboxes in xywh format to corners format."""
    return tf.concat(
        [
            bboxes[..., :2] - bboxes[..., 2:4] / 2.0,
            bboxes[..., :2] + bboxes[..., 2:4] / 2.0,
            bboxes[..., 4:],
        ],
        axis=-1,
    )


def pad_bbox_batch_to_shape(bboxes, target_shape, padding_values=-1):
    """Pads a list of bounding boxes with -1s.

    Boxes represented by all -1s are ignored by COCO metrics.

    Args:
        bboxes: tf.Tensor of bounding boxes in any format.
        target_shape: Target shape to pad bboxes to.
        padding_values: value to pad, defaults to -1 to mask out in coco metrics.
    Returns:
        bboxes padded to target shape.
    """
    bbox_shape = tf.shape(bboxes)
    paddings = [
        [0, target_shape - bbox_shape[i]]
        for (i, tartarget_shapeget) in enumerate(target_shape)
    ]
    return tf.pad(bboxes, paddings, mode="CONSTANT", constant_values=padding_values)
