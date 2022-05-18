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

"""Converter functions for working with bounding boxes.

Usually bounding boxes is a 2D Tensor with shape [batch, 4]. The second dimension
will contain 4 numbers based on 2 different formats.  In KerasCV, we will use the
`corners` format, which is [LEFT, TOP, RIGHT, BOTTOM].

In this file, provide utility functions for manipulating bounding boxes and converting
their formats.
"""

import tensorflow as tf


def _center_xywh_to_corners(boxes, images=None):
    x, y, width, height, rest = tf.split(boxes, [1, 1, 1, 1, -1], axis=-1)
    return tf.concat(
        [x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0, rest],
        axis=-1
    )


def _xywh_to_corners(boxes, images=None):
    x, y, width, height, rest = tf.split(boxes, [1, 1, 1, 1, -1], axis=-1)
    return tf.concat([x, y, x + width, y + height, rest], axis=-1)


def _xyxy_no_op(boxes, images=None):
    return boxes


def _corners_to_xywh(boxes, images=None):
    left, top, right, bottom, rest = tf.split(boxes, [1, 1, 1, 1, -1], axis=-1)
    return tf.concat([left, top, right - left, bottom - top, rest], axis=-1,)


def _corners_to_center_xywh(boxes, images=None):
    left, top, right, bottom, rest = tf.split(boxes, [1, 1, 1, 1, -1], axis=-1)
    return tf.concat(
        [(left + right) / 2.0, (top + bottom) / 2.0, right - left, bottom - top, rest],
        axis=-1
    )


to_corners_converters = {
    "xywh": _xywh_to_corners,
    "center_xywh": _center_xywh_to_corners,
    "xyxy": _xyxy_no_op,
}

from_corners_converters = {
    "xywh": _corners_to_xywh,
    "center_xywh": _corners_to_center_xywh,
    "xyxy": _xyxy_no_op,
}


def transform_format(boxes, source, target, images=None):
    """Converts bounding_boxes from one format to another.

    Supported formats are:
    - xyxy, also known as `corners` format.  In this format the first four axes
        represent [left, top, right, bottom] in that order.
    - xyWH.  In this format the first four axes represent [left, top, width, height].
    - center_xyWH.  In this format the first two coordinates represent the x and y
        coordinates of the center of the bounding box, while the last two represent
        the width and height of the bounding box.

    Formats are case insensitive.  It is recommended that you capitalize width and
    height to maximize the visual difference between `xyWH` and `xyxy`.

    Relative formats, abbreviated `rel`, make use of the shapes of the `images` passsed.
    In these formats, the coordinates, widths, and heights are all specified as
    percentages of the host image.  `images` may be a ragged Tensor.  Note that using a
    ragged Tensor for images may cause a substantial performance loss, as each image
    will need to be processed separately due to the mismatching image shapes.

    Usage:
    ```python
    boxes = load_coco_dataset()
    boxes_in_xywh = keras_cv.bounding_box.transform_format(
        boxes,
        original='xyxy',
        target='xyWH'
    )
    ```
    """
    global to_corners_converters
    global from_corners_converters
    source = source.lower()
    target = target.lower()
    if source not in to_corners_converters:
        raise ValueError(
            f"`transform_format()` received an unsupported format for the argument "
            f"`source`.  `source` should be one of {to_corners_converters.keys()}. "
            f"Got `source`={source}"
        )
    if target not in from_corners_converters:
        raise ValueError(
            f"`transform_format()` received an unsupported format for the argument "
            f"`target`.  `target` should be one of {from_corners_converters.keys()}. "
            f"Got `target`={target}"
        )

    to_corners = to_corners_converters[source]
    from_corners = from_corners_converters[target]

    in_corners = to_corners(boxes, images=images)
    return from_corners(in_corners, images=images)
