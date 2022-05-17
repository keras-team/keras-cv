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
