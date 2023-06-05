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

from keras_cv.models.backbones.csp_darknet import csp_darknet_backbone
from keras_cv.utils.train import get_feature_extractor


class CSPDarkNetBackboneTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.input_batch = np.ones(shape=(2, 224, 224, 3))

    def test_valid_call(self):
        model = csp_darknet_backbone.CSPDarkNetBackbone(
            stackwise_channels=[48, 96, 192, 384],
            stackwise_depth=[1, 3, 3, 1],
            include_rescaling=False,
        )
        model(self.input_batch)

    def test_valid_call_applications_model(self):
        model = csp_darknet_backbone.CSPDarkNetLBackbone()
        model(self.input_batch)

    def test_valid_call_with_rescaling(self):
        model = csp_darknet_backbone.CSPDarkNetBackbone(
            stackwise_channels=[48, 96, 192, 384],
            stackwise_depth=[1, 3, 3, 1],
            include_rescaling=True,
        )
        model(self.input_batch)

    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self):
        model = csp_darknet_backbone.CSPDarkNetBackbone(
            stackwise_channels=[48, 96, 192, 384],
            stackwise_depth=[1, 3, 3, 1],
            include_rescaling=True,
        )
        model_output = model(self.input_batch)
        save_path = os.path.join(
            self.get_temp_dir(), "csp_darknet_backbone.keras"
        )
        model.save(save_path)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(
            restored_model, csp_darknet_backbone.CSPDarkNetBackbone
        )

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)

    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_alias_model(self):
        model = csp_darknet_backbone.CSPDarkNetLBackbone()
        model_output = model(self.input_batch)
        save_path = os.path.join(
            self.get_temp_dir(), "csp_darknet_backbone.keras"
        )
        model.save(save_path)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        # Note that these aliases serialized as the base class
        self.assertIsInstance(
            restored_model, csp_darknet_backbone.CSPDarkNetBackbone
        )

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)

    def test_create_backbone_model_from_alias_model(self):
        model = csp_darknet_backbone.CSPDarkNetLBackbone(
            include_rescaling=False
        )
        backbone_model = get_feature_extractor(
            model,
            model.pyramid_level_inputs.values(),
            model.pyramid_level_inputs.keys(),
        )
        inputs = keras.Input(shape=[224, 224, 3])
        outputs = backbone_model(inputs)
        # CSPDarkNet backbone has 4 level of features (P2 ~ P5)
        levels = ["P2", "P3", "P4", "P5"]
        self.assertLen(outputs, 4)
        self.assertEquals(list(outputs.keys()), levels)
        self.assertEquals(outputs["P2"].shape, (None, 56, 56, 128))
        self.assertEquals(outputs["P3"].shape, (None, 28, 28, 256))
        self.assertEquals(outputs["P4"].shape, (None, 14, 14, 512))
        self.assertEquals(outputs["P5"].shape, (None, 7, 7, 1024))

    def test_create_backbone_model_with_level_config(self):
        model = csp_darknet_backbone.CSPDarkNetBackbone(
            stackwise_channels=[48, 96, 192, 384],
            stackwise_depth=[1, 3, 3, 1],
            include_rescaling=True,
        )
        levels = ["P3", "P4"]
        layer_names = [model.pyramid_level_inputs[level] for level in levels]
        backbone_model = get_feature_extractor(model, layer_names, levels)
        inputs = keras.Input(shape=[256, 256, 3])
        outputs = backbone_model(inputs)
        self.assertLen(outputs, 2)
        self.assertEquals(list(outputs.keys()), levels)
        self.assertEquals(outputs["P3"].shape, (None, 32, 32, 96))
        self.assertEquals(outputs["P4"].shape, (None, 16, 16, 192))

    @parameterized.named_parameters(
        ("Tiny", csp_darknet_backbone.CSPDarkNetTinyBackbone),
        ("S", csp_darknet_backbone.CSPDarkNetSBackbone),
        ("M", csp_darknet_backbone.CSPDarkNetMBackbone),
        ("L", csp_darknet_backbone.CSPDarkNetLBackbone),
        ("XL", csp_darknet_backbone.CSPDarkNetXLBackbone),
    )
    def test_specific_arch_forward_pass(self, arch_class):
        backbone = arch_class()
        backbone(np.random.uniform(size=(2, 256, 256, 3)))

    @parameterized.named_parameters(
        ("Tiny", csp_darknet_backbone.CSPDarkNetTinyBackbone),
        ("S", csp_darknet_backbone.CSPDarkNetSBackbone),
        ("M", csp_darknet_backbone.CSPDarkNetMBackbone),
        ("L", csp_darknet_backbone.CSPDarkNetLBackbone),
        ("XL", csp_darknet_backbone.CSPDarkNetXLBackbone),
    )
    def test_specific_arch_presets(self, arch_class):
        self.assertDictEqual(
            arch_class.presets, arch_class.presets_with_weights
        )

    def repro(self):
        model = keras.Sequential([keras.layers.Dense(10), keras.layers.Lambda(lambda x: keras.activations.silu(x))])
        x = keras.layers.Input(shape=(224, 224, 3))
        x = model(x)


if __name__ == "__main__":
    tf.test.main()
