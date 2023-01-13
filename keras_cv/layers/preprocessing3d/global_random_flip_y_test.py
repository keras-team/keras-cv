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
import numpy as np
import tensorflow as tf

from keras_cv.layers.preprocessing3d import base_augmentation_layer_3d
from keras_cv.layers.preprocessing3d.global_random_flip_y import GlobalRandomFlipY

POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES


class GlobalFlippingYTest(tf.test.TestCase):
    def test_augment_random_point_clouds_and_bounding_boxes(self):
        add_layer = GlobalRandomFlipY()
        point_clouds = np.random.random(size=(2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertNotAllClose(inputs, outputs)

    def test_augment_specific_random_point_clouds_and_bounding_boxes(self):
        add_layer = GlobalRandomFlipY()
        point_clouds = np.array([[[1, 1, 2, 3, 4, 5, 6, 7, 8, 9]] * 2] * 2).astype(
            "float32"
        )
        bounding_boxes = np.array([[[1, 1, 2, 3, 4, 5, 1]] * 2] * 2).astype("float32")

        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        flipped_point_clouds = np.array(
            [[[1, -1, 2, 3, 4, 5, 6, 7, 8, 9]] * 2] * 2
        ).astype("float32")
        flipped_bounding_boxes = np.array([[[1, -1, 2, 3, 4, 5, -1]] * 2] * 2).astype(
            "float32"
        )
        self.assertAllClose(outputs[POINT_CLOUDS], flipped_point_clouds)
        self.assertAllClose(outputs[BOUNDING_BOXES], flipped_bounding_boxes)

    def test_augment_batch_point_clouds_and_bounding_boxes(self):
        add_layer = GlobalRandomFlipY()
        point_clouds = np.random.random(size=(3, 2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(3, 2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertNotAllClose(inputs, outputs)
