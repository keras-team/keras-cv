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
from keras_cv.layers.preprocessing3d.frustum_random_point_feature_noise import (
    FrustumRandomPointFeatureNoise,
)

POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES
POINTCLOUD_LABEL_INDEX = base_augmentation_layer_3d.POINTCLOUD_LABEL_INDEX


class FrustumRandomPointFeatureNoiseTest(tf.test.TestCase):
    def test_augment_point_clouds_and_bounding_boxes(self):
        add_layer = FrustumRandomPointFeatureNoise(
            r_distance=0, theta_width=1, phi_width=1, max_noise_level=0.5
        )
        point_clouds = np.random.random(size=(2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertNotAllClose(inputs, outputs)
        # bounding boxes and point clouds (x, y, z, class) are not modified.
        self.assertAllClose(inputs[BOUNDING_BOXES], outputs[BOUNDING_BOXES])
        self.assertAllClose(
            inputs[POINT_CLOUDS][:, :, :POINTCLOUD_LABEL_INDEX],
            outputs[POINT_CLOUDS][:, :, :POINTCLOUD_LABEL_INDEX],
        )

    def test_augment_specific_point_clouds_and_bounding_boxes(self):
        tf.keras.utils.set_random_seed(2)
        add_layer = FrustumRandomPointFeatureNoise(
            r_distance=10, theta_width=np.pi, phi_width=1.5 * np.pi, max_noise_level=0.5
        )
        point_clouds = np.array(
            [
                [
                    [0, 1, 2, 3, 4, 5],
                    [10, 1, 2, 3, 4, 2],
                    [100, 100, 2, 3, 4, 1],
                    [-20, -20, 21, 1, 0, 2],
                ]
            ]
            * 2
        ).astype("float32")
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        # bounding boxes and point clouds (x, y, z, class) are not modified.
        augmented_point_clouds = np.array(
            [
                [
                    [0, 1, 2, 3, 4, 5],
                    [10, 1, 2, 3, 4, 2],
                    [100, 100, 2, 3, 4, 1],
                    [-20, -20, 21, 1, 0, 1.3747642],
                ],
                [
                    [0, 1, 2, 3, 4, 5],
                    [10, 1, 2, 3, 4, 2],
                    [100, 100, 2, 3, 4, 1],
                    [-20, -20, 21, 1, 0, 1.6563809],
                ],
            ]
        ).astype("float32")
        self.assertAllClose(inputs[BOUNDING_BOXES], outputs[BOUNDING_BOXES])
        # [-20, -20, 21, 1, 0, 2] is randomly selected as the frustum center.
        # [0, 1, 2, 3, 4, 5] and [10, 1, 2, 3, 4, 2] are not changed due to less than r_distance.
        # [100, 100, 2, 3, 4, 1] is not changed due to outside phi_width.
        self.assertAllClose(outputs[POINT_CLOUDS], augmented_point_clouds)

    def test_augment_only_one_valid_point_point_clouds_and_bounding_boxes(self):
        tf.keras.utils.set_random_seed(2)
        add_layer = FrustumRandomPointFeatureNoise(
            r_distance=10, theta_width=np.pi, phi_width=1.5 * np.pi, max_noise_level=0.5
        )
        point_clouds = np.array(
            [
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [100, 100, 2, 3, 4, 1],
                    [0, 0, 0, 0, 0, 0],
                ]
            ]
            * 2
        ).astype("float32")
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        # bounding boxes and point clouds (x, y, z, class) are not modified.
        augmented_point_clouds = np.array(
            [
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [100, 100, 2, 3, 4.119616, 0.619783],
                    [0, 0, 0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [100, 100, 2, 3, 3.192014, 0.618371],
                    [0, 0, 0, 0, 0, 0],
                ],
            ]
        ).astype("float32")
        self.assertAllClose(inputs[BOUNDING_BOXES], outputs[BOUNDING_BOXES])
        # [100, 100, 2, 3, 4, 1] is selected as the frustum center because it is the only valid point.
        self.assertAllClose(outputs[POINT_CLOUDS], augmented_point_clouds)

    def test_not_augment_max_noise_level0_point_clouds_and_bounding_boxes(self):
        add_layer = FrustumRandomPointFeatureNoise(
            r_distance=0, theta_width=1, phi_width=1, max_noise_level=0.0
        )
        point_clouds = np.random.random(size=(2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertAllClose(inputs, outputs)

    def test_not_augment_max_noise_level1_frustum_empty_point_clouds_and_bounding_boxes(
        self,
    ):
        add_layer = FrustumRandomPointFeatureNoise(
            r_distance=10, theta_width=0, phi_width=0, max_noise_level=1.0
        )
        point_clouds = np.random.random(size=(2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertAllClose(inputs, outputs)

    def test_augment_batch_point_clouds_and_bounding_boxes(self):
        add_layer = FrustumRandomPointFeatureNoise(
            r_distance=0, theta_width=1, phi_width=1, max_noise_level=0.5
        )
        point_clouds = np.random.random(size=(3, 2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(3, 2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertNotAllClose(inputs, outputs)
