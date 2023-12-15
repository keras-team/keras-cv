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
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_cv.backend.config import keras_3
from keras_cv.layers import preprocessing_3d
from keras_cv.layers.preprocessing_3d import base_augmentation_layer_3d
from keras_cv.tests.test_case import TestCase

if not keras_3():
    POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
    BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES

    TEST_CONFIGURATIONS = [
        (
            "FrustrumRandomDroppingPoints",
            preprocessing_3d.FrustumRandomDroppingPoints(
                r_distance=0, theta_width=1, phi_width=1, drop_rate=0.5
            ),
        ),
        (
            "FrustrumRandomPointFeatureNoise",
            preprocessing_3d.FrustumRandomPointFeatureNoise(
                r_distance=10,
                theta_width=np.pi,
                phi_width=1.5 * np.pi,
                max_noise_level=0.5,
            ),
        ),
        (
            "GlobalRandomDroppingPoints",
            preprocessing_3d.GlobalRandomDroppingPoints(drop_rate=0.5),
        ),
        (
            "GlobalRandomFlip",
            preprocessing_3d.GlobalRandomFlip(),
        ),
        (
            "GlobalRandomRotation",
            preprocessing_3d.GlobalRandomRotation(
                max_rotation_angle_x=1.0,
                max_rotation_angle_y=1.0,
                max_rotation_angle_z=1.0,
            ),
        ),
        (
            "GlobalRandomScaling",
            preprocessing_3d.GlobalRandomScaling(
                x_factor=(0.5, 1.5),
                y_factor=(0.5, 1.5),
                z_factor=(0.5, 1.5),
            ),
        ),
        (
            "GlobalRandomTranslation",
            preprocessing_3d.GlobalRandomTranslation(
                x_stddev=1.0, y_stddev=1.0, z_stddev=1.0
            ),
        ),
        (
            "RandomDropBox",
            preprocessing_3d.RandomDropBox(
                label_index=1, max_drop_bounding_boxes=4
            ),
        ),
    ]

    def convert_to_model_format(inputs):
        point_clouds = {
            "point_xyz": inputs["point_clouds"][..., :3],
            "point_feature": inputs["point_clouds"][..., 3:-1],
            "point_mask": tf.cast(inputs["point_clouds"][..., -1], tf.bool),
        }
        boxes = {
            "boxes": inputs["bounding_boxes"][..., :7],
            "classes": inputs["bounding_boxes"][..., 7],
            "difficulty": inputs["bounding_boxes"][..., -1],
            "mask": tf.cast(inputs["bounding_boxes"][..., 8], tf.bool),
        }
        return {
            "point_clouds": point_clouds,
            "3d_boxes": boxes,
        }

    @pytest.skip(
        reason="values are not matching because of changes to random.py"
    )
    class InputFormatTest(TestCase):
        @parameterized.named_parameters(*TEST_CONFIGURATIONS)
        def test_equivalent_results_with_model_format(self, layer):
            point_clouds = np.random.random(size=(3, 2, 50, 10)).astype(
                "float32"
            )
            bounding_boxes = np.random.random(size=(3, 2, 10, 9)).astype(
                "float32"
            )
            inputs = {
                POINT_CLOUDS: point_clouds,
                BOUNDING_BOXES: bounding_boxes,
            }

            tf.random.set_seed(123)
            outputs_with_legacy_format = convert_to_model_format(layer(inputs))
            tf.random.set_seed(123)
            outputs_with_model_format = layer(convert_to_model_format(inputs))

            self.assertAllClose(
                outputs_with_legacy_format, outputs_with_model_format
            )
