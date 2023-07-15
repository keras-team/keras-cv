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

from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX002Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX004Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX006Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX008Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX016Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX032Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX040Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX064Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX080Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX120Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX160Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX320Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY002Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY004Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY006Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY008Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY016Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY032Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY040Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY064Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY080Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY120Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY160Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY320Backbone
from keras_cv.models.backbones.regnet.regnet_backbone import RegNetBackbone
from keras_cv.tests.test_case import TestCase
from keras_cv.utils.train import get_feature_extractor


class RegNetBackboneTest(TestCase):
    def setUp(self):
        self.input_batch = np.ones(shape=(2, 224, 224, 3))

    def test_valid_call(self):
        model = RegNetBackbone(
            depths=[1, 1, 4, 7],
            widths=[24, 56, 152, 368],
            group_width=8,
            include_rescaling=False,
            block_type="X",
        )
        model(self.input_batch)

    def test_valid_call_applications_model(self):
        model = RegNetX002Backbone()
        model(self.input_batch)

    def test_valid_call_with_rescaling(self):
        model = RegNetBackbone(
            depths=[1, 1, 4, 7],
            widths=[24, 56, 152, 368],
            group_width=8,
            include_rescaling=True,
            block_type="X"
        )
        model(self.input_batch)

    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self):
        model = RegNetBackbone(
            depths=[1, 1, 4, 7],
            widths=[24, 56, 152, 368],
            group_width=8,
            include_rescaling=False,
            block_type="X",
        )
        model_output = model(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(), "regnet_backbone.keras")
        model.save(save_path)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, RegNetBackbone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)

    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_alias_model(self):
        model = RegNetX002Backbone()
        model_output = model(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(), "regnet_backbone.keras")
        model.save(save_path)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        # Note that these aliases serialized as the base class
        self.assertIsInstance(restored_model, RegNetBackbone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)

    def test_model_backbone_layer_names_stability(self):
        model = RegNetBackbone(
            depths=[1, 1, 4, 7],
            widths=[24, 56, 152, 368],
            group_width=8,
            include_rescaling=False,
            block_type="X",
        )
        model_2 = RegNetBackbone(
            depths=[1, 1, 4, 7],
            widths=[24, 56, 152, 368],
            group_width=8,
            include_rescaling=False,
            block_type="X",
        )
        layers_1 = model.layers
        layers_2 = model_2.layers
        for i in range(len(layers_1)):
            if "input" in layers_1[i].name:
                continue
            self.assertEquals(layers_1[i].name, layers_2[i].name)

    def test_feature_pyramid_inputs(self):
        model = RegNetX002Backbone()
        backbone_model = get_feature_extractor(
            model,
            model.pyramid_level_inputs.values(),
            model.pyramid_level_inputs.keys(),
        )
        input_size = 256
        inputs = keras.Input(shape=[input_size, input_size, 3])
        outputs = backbone_model(inputs)
        levels = ["P1"]
        self.assertEquals(list(outputs.keys()), levels)
        self.assertEquals(
            outputs["P1"].shape,
            (None, 8, 8, 368),
        )

    @parameterized.named_parameters(
        ("one_channel", 1),
        ("four_channels", 4),
    )
    def test_application_variable_input_channels(self, num_channels):
        # RegNetX002 model
        model = RegNetBackbone(
            depths=[1, 1, 4, 7],
            widths=[24, 56, 152, 368],
            group_width=8,
            input_shape=(None, None, num_channels),
            include_rescaling=False,
            block_type="X",
        )
        self.assertEqual(model.output_shape, (None, None, None, 368))

    @parameterized.named_parameters(
        ("x002", RegNetX002Backbone),
        ("x004", RegNetX004Backbone),
        ("x006", RegNetX006Backbone),
        ("x008", RegNetX008Backbone),
        ("x016", RegNetX016Backbone),
        ("x032", RegNetX032Backbone),
        ("x040", RegNetX040Backbone),
        ("x064", RegNetX064Backbone),
        ("x080", RegNetX080Backbone),
        ("x120", RegNetX120Backbone),
        ("x160", RegNetX160Backbone),
        ("x320", RegNetX320Backbone),
        ("y002", RegNetY002Backbone),
        ("y004", RegNetY004Backbone),
        ("y006", RegNetY006Backbone),
        ("y008", RegNetY008Backbone),
        ("y016", RegNetY016Backbone),
        ("y032", RegNetY032Backbone),
        ("y040", RegNetY040Backbone),
        ("y064", RegNetY064Backbone),
        ("y080", RegNetY080Backbone),
        ("y120", RegNetY120Backbone),
        ("y160", RegNetY160Backbone),
        ("y320", RegNetY320Backbone),
    )
    def test_specific_arch_forward_pass(self, arch_class):
        backbone = arch_class()
        backbone(np.random.uniform(size=[2, 256, 256, 3]))
