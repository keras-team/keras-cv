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

import numpy as np
import pytest
from absl.testing import parameterized

from keras_cv.backend import keras
from keras_cv.models.backbones.regnet.regnet_aliases import (  # noqa: E501
    RegNetX002Backbone,
)
from keras_cv.models.backbones.regnet.regnet_backbone import (  # noqa: E501
    RegNetBackbone,
)
from keras_cv.tests.test_case import TestCase
from keras_cv.utils.train import get_feature_extractor


@pytest.mark.extra_large
class RegNetPresetFullTest(TestCase):
    """
    Test the full enumeration of our preset.
    This tests every preset for EfficientNetLite and is only run manually.
    Run with:
    `pytest keras_cv/models/backbones/regnet/regnet_backbone_presets_test.py --run_extra_large`
    """  # noqa: E501

    @parameterized.named_parameters(
        *[(preset, preset) for preset in RegNetBackbone.presets]
    )
    def test_load_efficientnetlite(self, preset):
        input_data = np.ones(shape=(2, 224, 224, 3))
        model = RegNetBackbone.from_preset(preset)
        model(input_data)

    def test_efficientnetlite_feature_extractor(self):
        model = RegNetX002Backbone(
            include_rescaling=False,
            input_shape=[256, 256, 3],
        )
        levels = ["P1"]
        layer_names = [model.pyramid_level_inputs[level] for level in levels]
        backbone_model = get_feature_extractor(model, layer_names, levels)
        inputs = keras.Input(shape=[256, 256, 3])
        outputs = backbone_model(inputs)
        self.assertLen(outputs, 1)
        self.assertEquals(list(outputs.keys()), levels)
        self.assertEquals(outputs["P1"].shape[:3], (None, 8, 8))
