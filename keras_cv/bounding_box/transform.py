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
"""
transform.py contains function to transform bounding box according to corner transformation
"""

import tensorflow as tf

import keras_cv.bounding_box as bounding_box


def transform_from_point_transform(
    bounding_boxes,
    point_transform_fn,
    bounding_box_format="xyxy",
    images=None,
    dtype=None,
    clip_boxes=True,
):
    """Transform bounding boxes by applying a function to each of their 4 corners.

    Args:
      bounding_boxes: a 2D or 3D or 4D tensor like value of bounding boxes.
      point_transform_fn: a function that transform a list of points.
      bounding_box_format: the input format of the bounding boxes.
      images: the images tensor used to access the image shape for
        relative/absolute coordinate transforms.
      dtype: the datatype to return value, defaults to bounding_box dtype if None.
      clip_boxes: clip the boxes to the image size.

    Returns:
      A tensor of transformed bounding boxes.
    """
    dtype = dtype or bounding_boxes.dtype

    if clip_boxes and images is None:
        raise ValueError(
            "transform_from_point_transform() called with "
            "`clip_boxes`=True, but no image tensor passed"
        )

    bounding_boxes = bounding_box.convert_format(
        bounding_boxes,
        source=bounding_box_format,
        target="xyxy",
        images=images,
    )
    xlu, ylu, xrl, yrl, rest = tf.split(bounding_boxes, [1, 1, 1, 1, -1], axis=-1)
    corners = tf.stack(
        [
            tf.concat([xlu, ylu], axis=1),
            tf.concat([xrl, ylu], axis=1),
            tf.concat([xrl, yrl], axis=1),
            tf.concat([xlu, yrl], axis=1),
        ],
        axis=1,
    )
    corners = point_transform_fn(corners, )
    min_coordinates = tf.math.reduce_min(corners, axis=-2)
    max_coordinates = tf.math.reduce_max(corners, axis=-2)

    if clip_boxes:
        image_shape = tf.cast(tf.shape(images), max_coordinates.dtype)
        min_coordinates = tf.maximum(min_coordinates, 0.0)
        max_coordinates = tf.minimum(max_coordinates, image_shape[-3:-1])

    bounding_boxes_out = tf.concat([min_coordinates, max_coordinates, rest], axis=-1)
    bounding_boxes_out = bounding_box.convert_format(
        bounding_boxes_out,
        source="xyxy",
        target=bounding_box_format,
        images=images,
        dtype=dtype,
    )
    return bounding_boxes_out
