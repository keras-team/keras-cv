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
from keras_cv.layers.preprocessing3d.global_random_scaling import GlobalRandomScaling

POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES


class GlobalScalingTest(tf.test.TestCase):
    def test_augment_point_clouds_and_bounding_boxes(self):
        add_layer = GlobalRandomScaling(
            scaling_factor_x=(0.5, 1.5),
            scaling_factor_y=(0.5, 1.5),
            scaling_factor_z=(0.5, 1.5),
        )
        point_clouds = np.random.random(size=(2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertNotAllClose(inputs, outputs)

    def test_augment_point_clouds_and_bounding_boxes_with_same_scaling(self):
        add_layer = GlobalRandomScaling(
            scaling_factor_x=(0.5, 1.5),
            scaling_factor_y=(0.5, 1.5),
            scaling_factor_z=(0.5, 1.5),
            same_scaling_xyz=True,
        )
        point_clouds = np.random.random(size=(2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertNotAllClose(inputs, outputs)

    def test_not_augment_point_clouds_and_bounding_boxes(self):
        add_layer = GlobalRandomScaling(
            scaling_factor_x=(1.0, 1.0),
            scaling_factor_y=(1.0, 1.0),
            scaling_factor_z=(1.0, 1.0),
        )
        point_clouds = np.random.random(size=(2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertAllClose(inputs, outputs)

    def test_2x_scaling_point_clouds_and_bounding_boxes(self):
        add_layer = GlobalRandomScaling(
            scaling_factor_x=(2.0, 2.0),
            scaling_factor_y=(2.0, 2.0),
            scaling_factor_z=(2.0, 2.0),
        )
        point_clouds = np.array([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] * 2] * 2).astype(
            "float32"
        )
        bounding_boxes = np.array([[[0, 1, 2, 3, 4, 5, 6]] * 2] * 2).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        scaled_point_clouds = np.array(
            [[[0, 2, 4, 3, 4, 5, 6, 7, 8, 9]] * 2] * 2
        ).astype("float32")
        scaled_bounding_boxes = np.array([[[0, 2, 4, 6, 8, 10, 6]] * 2] * 2).astype(
            "float32"
        )
        self.assertAllClose(outputs[POINT_CLOUDS], scaled_point_clouds)
        self.assertAllClose(outputs[BOUNDING_BOXES], scaled_bounding_boxes)

    def test_augment_batch_point_clouds_and_bounding_boxes(self):
        add_layer = GlobalRandomScaling(
            scaling_factor_x=(0.5, 1.5),
            scaling_factor_y=(0.5, 1.5),
            scaling_factor_z=(0.5, 1.5),
        )
        point_clouds = np.random.random(size=(3, 2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(3, 2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertNotAllClose(inputs, outputs)

    def test_not_augment_batch_point_clouds_and_bounding_boxes(self):
        add_layer = GlobalRandomScaling(
            scaling_factor_x=(1.0, 1.0),
            scaling_factor_y=(1.0, 1.0),
            scaling_factor_z=(1.0, 1.0),
        )
        point_clouds = np.random.random(size=(3, 2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(3, 2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertAllClose(inputs, outputs)
