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
from keras_cv.layers.preprocessing3d.global_random_dropping_points import (
    GlobalRandomDroppingPoints,
)

POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES


class GlobalDropPointsTest(tf.test.TestCase):
    def test_augment_point_clouds_and_bounding_boxes(self):
        add_layer = GlobalRandomDroppingPoints(drop_rate=0.5)

        point_clouds = np.random.random(size=(2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertNotAllClose(inputs, outputs)

    def test_specific_augment_point_clouds_and_bounding_boxes(self):
        add_layer = GlobalRandomDroppingPoints(drop_rate=0.5)

        point_clouds = np.random.random(size=(1, 50, 2)).astype("float32")
        point_clouds = np.concatenate([point_clouds, point_clouds], axis=0)
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertNotAllClose(inputs, outputs)
        # The augmented point clouds in the first frame should be the same as the
        # augmented point clouds in the second frame.
        self.assertAllClose(outputs[POINT_CLOUDS][0], outputs[POINT_CLOUDS][1])

    def test_not_augment_point_clouds_and_bounding_boxes(self):
        add_layer = GlobalRandomDroppingPoints(drop_rate=0.0)

        point_clouds = np.random.random(size=(2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertAllClose(inputs, outputs)

    def test_drop_all_point_clouds(self):
        add_layer = GlobalRandomDroppingPoints(drop_rate=1.0)

        point_clouds = np.random.random(size=(2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertAllClose(inputs[POINT_CLOUDS] * 0.0, outputs[POINT_CLOUDS])

    def test_augment_batch_point_clouds_and_bounding_boxes(self):
        add_layer = GlobalRandomDroppingPoints(drop_rate=0.5)

        point_clouds = np.random.random(size=(3, 2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(3, 2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertNotAllClose(inputs, outputs)
