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
from absl.testing import parameterized

from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_aliases import (
    EfficientNetV2SBackbone,
)
from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_backbone import (
    EfficientNetV2Backbone,
)
from keras_cv.tests.test_case import TestCase
from keras_cv.utils.train import get_feature_extractor


class EfficientNetV2BackboneTest(TestCase):
    def setUp(self):
        self.input_batch = np.ones(shape=(8, 224, 224, 3))

    def test_valid_call(self):
        model = EfficientNetV2Backbone(
            stackwise_kernel_sizes=[3, 3, 3, 3, 3, 3],
            stackwise_num_repeats=[2, 4, 4, 6, 9, 15],
            stackwise_input_filters=[24, 24, 48, 64, 128, 160],
            stackwise_output_filters=[24, 48, 64, 128, 160, 256],
            stackwise_expansion_ratios=[1, 4, 4, 4, 6, 6],
            stackwise_squeeze_and_excite_ratios=[0.0, 0.0, 0, 0.25, 0.25, 0.25],
            stackwise_strides=[1, 2, 2, 2, 1, 2],
            stackwise_conv_types=[
                "fused",
                "fused",
                "fused",
                "unfused",
                "unfused",
                "unfused",
            ],
            width_coefficient=1.0,
            depth_coefficient=1.0,
            include_rescaling=False,
        )
        model(self.input_batch)

    def test_valid_call_alias_model_with_rescaling(self):
        model = EfficientNetV2SBackbone(include_rescaling=True)
        model(self.input_batch)

    def test_valid_call_with_rescaling(self):
        model = EfficientNetV2Backbone(
            stackwise_kernel_sizes=[3, 3, 3, 3, 3, 3],
            stackwise_num_repeats=[2, 4, 4, 6, 9, 15],
            stackwise_input_filters=[24, 24, 48, 64, 128, 160],
            stackwise_output_filters=[24, 48, 64, 128, 160, 256],
            stackwise_expansion_ratios=[1, 4, 4, 4, 6, 6],
            stackwise_squeeze_and_excite_ratios=[0.0, 0.0, 0, 0.25, 0.25, 0.25],
            stackwise_strides=[1, 2, 2, 2, 1, 2],
            stackwise_conv_types=[
                "fused",
                "fused",
                "fused",
                "unfused",
                "unfused",
                "unfused",
            ],
            width_coefficient=1.0,
            depth_coefficient=1.0,
            include_rescaling=True,
        )
        model(self.input_batch)

    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self):
        model = EfficientNetV2Backbone(
            stackwise_kernel_sizes=[3, 3, 3, 3, 3, 3],
            stackwise_num_repeats=[2, 4, 4, 6, 9, 15],
            stackwise_input_filters=[24, 24, 48, 64, 128, 160],
            stackwise_output_filters=[24, 48, 64, 128, 160, 256],
            stackwise_expansion_ratios=[1, 4, 4, 4, 6, 6],
            stackwise_squeeze_and_excite_ratios=[0.0, 0.0, 0, 0.25, 0.25, 0.25],
            stackwise_strides=[1, 2, 2, 2, 1, 2],
            stackwise_conv_types=[
                "fused",
                "fused",
                "fused",
                "unfused",
                "unfused",
                "unfused",
            ],
            width_coefficient=1.0,
            depth_coefficient=1.0,
            include_rescaling=True,
        )
        model_output = model(self.input_batch)
        save_path = os.path.join(
            self.get_temp_dir(), "efficientnet_v2_backbone.keras"
        )
        model.save(save_path)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, EfficientNetV2Backbone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(
            ops.convert_to_numpy(model_output),
            ops.convert_to_numpy(restored_output),
        )

    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_alias_model(self):
        model = EfficientNetV2SBackbone()
        model_output = model(self.input_batch)
        save_path = os.path.join(
            self.get_temp_dir(), "efficientnet_v2_backbone.keras"
        )
        model.save(save_path)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        # Note that these aliases serialized as the base class
        self.assertIsInstance(restored_model, EfficientNetV2Backbone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(
            ops.convert_to_numpy(model_output),
            ops.convert_to_numpy(restored_output),
        )

    def test_feature_pyramid_inputs(self):
        model = EfficientNetV2SBackbone()
        backbone_model = get_feature_extractor(
            model,
            model.pyramid_level_inputs.values(),
            model.pyramid_level_inputs.keys(),
        )
        input_size = 256
        inputs = keras.Input(shape=[input_size, input_size, 3])
        outputs = backbone_model(inputs)
        levels = ["P1", "P2", "P3", "P4", "P5"]
        self.assertEquals(list(outputs.keys()), levels)
        self.assertEquals(
            outputs["P1"].shape,
            (None, input_size // 2**1, input_size // 2**1, 24),
        )
        self.assertEquals(
            outputs["P2"].shape,
            (None, input_size // 2**2, input_size // 2**2, 48),
        )
        self.assertEquals(
            outputs["P3"].shape,
            (None, input_size // 2**3, input_size // 2**3, 64),
        )
        self.assertEquals(
            outputs["P4"].shape,
            (None, input_size // 2**4, input_size // 2**4, 160),
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
        model = EfficientNetV2Backbone(
            stackwise_kernel_sizes=[3, 3, 3, 3, 3, 3],
            stackwise_num_repeats=[2, 4, 4, 6, 9, 15],
            stackwise_input_filters=[24, 24, 48, 64, 128, 160],
            stackwise_output_filters=[24, 48, 64, 128, 160, 256],
            stackwise_expansion_ratios=[1, 4, 4, 4, 6, 6],
            stackwise_squeeze_and_excite_ratios=[0.0, 0.0, 0, 0.25, 0.25, 0.25],
            stackwise_strides=[1, 2, 2, 2, 1, 2],
            stackwise_conv_types=[
                "fused",
                "fused",
                "fused",
                "unfused",
                "unfused",
                "unfused",
            ],
            width_coefficient=1.0,
            depth_coefficient=1.0,
            include_rescaling=True,
        )
        self.assertEqual(model.output_shape, (None, None, None, 1280))
