# Copyright 2023 The KerasCV Authors
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
import tensorflow as tf

from keras_cv import bounding_box


def get_anchors(
    image_shape=(512, 512, 3),
    strides=[8, 16, 32],
    base_anchors=[-0.5, -0.5, 0.5, 0.5],
):
    base_anchors = tf.constant(base_anchors, dtype=tf.float32)

    all_anchors = []
    for stride in strides:
        top, left = (stride / 2, stride / 2)
        hh_centers = tf.range(top, image_shape[0], stride)
        ww_centers = tf.range(left, image_shape[1], stride)
        ww_grid, hh_grid = tf.meshgrid(ww_centers, hh_centers)
        grid = tf.cast(
            tf.reshape(
                tf.stack([hh_grid, ww_grid, hh_grid, ww_grid], 2), [-1, 1, 4]
            ),
            tf.float32,
        )
        anchors = (
            tf.expand_dims(base_anchors * [stride, stride, stride, stride], 0)
            + grid
        )
        anchors = tf.reshape(anchors, [-1, 4])
        all_anchors.append(anchors)

    all_anchors = tf.concat(all_anchors, axis=0)
    all_anchors = bounding_box.convert_format(
        all_anchors,
        source="yxyx",
        target="rel_yxyx",
        image_shape=list(image_shape),
    )

    return tf.cast(all_anchors, tf.float32)
