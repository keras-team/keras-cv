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
    compute_dtype=tf.float32,
):
    bounding_boxes = bounding_box.convert_format(
        bounding_boxes, source=bounding_box_format, target="xyxy"
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
    corners = point_transform_fn(corners)
    min_coordinates = tf.math.reduce_min(corners, axis=-2)
    max_coordinates = tf.math.reduce_max(corners, axis=-2)
    bounding_boxes_out = tf.concat([min_coordinates, max_coordinates, rest], axis=-1)
    bounding_boxes_out = bounding_box.convert_format(
        bounding_boxes_out,
        source="xyxy",
        target=bounding_box_format,
        dtype=compute_dtype,
    )
    return bounding_boxes_out
