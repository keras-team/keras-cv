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

import os

import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_cv.backend import keras
from keras_cv.models.backbones.efficientnet_v1.efficientnet_v1_aliases import (
    EfficientNetV1B0Backbone,
)
from keras_cv.models.backbones.efficientnet_v1.efficientnet_v1_backbone import (
    EfficientNetV1Backbone,
)
from keras_cv.tests.test_case import TestCase
from keras_cv.utils.train import get_feature_extractor


class EfficientNetV1BackboneTest(TestCase):
    def setUp(self):
        self.input_batch = np.ones(shape=(8, 224, 224, 3))

    def test_valid_call(self):
        model = EfficientNetV1Backbone(
            stackwise_kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
            stackwise_num_repeats=[1, 2, 2, 3, 3, 4, 1],
            stackwise_input_filters=[32, 16, 24, 40, 80, 112, 192],
            stackwise_output_filters=[16, 24, 40, 80, 112, 192, 320],
            stackwise_expansion_ratios=[1, 6, 6, 6, 6, 6, 6],
            stackwise_strides=[1, 2, 2, 2, 1, 2, 1],
            stackwise_squeeze_and_excite_ratios=[
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
            ],
            width_coefficient=1.0,
            depth_coefficient=1.0,
            include_rescaling=False,
        )
        model(self.input_batch)

    def test_valid_call_alias_model_with_rescaling(self):
        model = EfficientNetV1B0Backbone(include_rescaling=True)
        model(self.input_batch)

    def test_valid_call_with_rescaling(self):
        model = EfficientNetV1Backbone(
            stackwise_kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
            stackwise_num_repeats=[1, 2, 2, 3, 3, 4, 1],
            stackwise_input_filters=[32, 16, 24, 40, 80, 112, 192],
            stackwise_output_filters=[16, 24, 40, 80, 112, 192, 320],
            stackwise_expansion_ratios=[1, 6, 6, 6, 6, 6, 6],
            stackwise_strides=[1, 2, 2, 2, 1, 2, 1],
            stackwise_squeeze_and_excite_ratios=[
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
            ],
            width_coefficient=1.0,
            depth_coefficient=1.0,
            include_rescaling=True,
        )
        model(self.input_batch)

    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self):
        model = EfficientNetV1Backbone(
            stackwise_kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
            stackwise_num_repeats=[1, 2, 2, 3, 3, 4, 1],
            stackwise_input_filters=[32, 16, 24, 40, 80, 112, 192],
            stackwise_output_filters=[16, 24, 40, 80, 112, 192, 320],
            stackwise_expansion_ratios=[1, 6, 6, 6, 6, 6, 6],
            stackwise_strides=[1, 2, 2, 2, 1, 2, 1],
            stackwise_squeeze_and_excite_ratios=[
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
            ],
            width_coefficient=1.0,
            depth_coefficient=1.0,
            include_rescaling=True,
        )
        model_output = model(self.input_batch)
        save_path = os.path.join(
            self.get_temp_dir(), "efficientnet_v1_backbone.keras"
        )
        model.save(save_path)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, EfficientNetV1Backbone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)

    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_alias_model(self):
        model = EfficientNetV1B0Backbone()
        model_output = model(self.input_batch)
        save_path = os.path.join(
            self.get_temp_dir(), "efficientnet_v1_backbone.keras"
        )
        model.save(save_path)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        # Note that these aliases serialized as the base class
        self.assertIsInstance(restored_model, EfficientNetV1Backbone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)

    def test_feature_pyramid_inputs(self):
        model = EfficientNetV1B0Backbone()
        backbone_model = get_feature_extractor(
            model,
            model.pyramid_level_inputs.values(),
            model.pyramid_level_inputs.keys(),
        )
        input_size = 256
        inputs = tf.keras.Input(shape=[input_size, input_size, 3])
        outputs = backbone_model(inputs)
        levels = ["P1", "P2", "P3", "P4", "P5"]
        self.assertEquals(list(outputs.keys()), levels)
        self.assertEquals(
            outputs["P1"].shape,
            (None, input_size // 2**1, input_size // 2**1, 16),
        )
        self.assertEquals(
            outputs["P2"].shape,
            (None, input_size // 2**2, input_size // 2**2, 24),
        )
        self.assertEquals(
            outputs["P3"].shape,
            (None, input_size // 2**3, input_size // 2**3, 40),
        )
        self.assertEquals(
            outputs["P4"].shape,
            (None, input_size // 2**4, input_size // 2**4, 112),
        )
        self.assertEquals(
            outputs["P5"].shape,
            (None, input_size // 2**5, input_size // 2**5, 1280),
        )

    @parameterized.named_parameters(
        ("one_channel", 1),
        ("four_channels", 4),
    )
    def test_application_variable_input_channels(self, num_channels):
        model = EfficientNetV1Backbone(
            stackwise_kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
            stackwise_num_repeats=[1, 2, 2, 3, 3, 4, 1],
            stackwise_input_filters=[32, 16, 24, 40, 80, 112, 192],
            stackwise_output_filters=[16, 24, 40, 80, 112, 192, 320],
            stackwise_expansion_ratios=[1, 6, 6, 6, 6, 6, 6],
            stackwise_strides=[1, 2, 2, 2, 1, 2, 1],
            stackwise_squeeze_and_excite_ratios=[
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
            ],
            width_coefficient=1.0,
            depth_coefficient=1.0,
            include_rescaling=True,
        )
        self.assertEqual(model.output_shape, (None, None, None, 1280))
