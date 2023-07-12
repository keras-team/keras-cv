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
from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_aliases import (
    EfficientNetV2SBackbone,
)
from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_backbone import (
    EfficientNetV2Backbone,
)
from keras_cv.tests.test_case import TestCase
from keras_cv.utils.train import get_feature_extractor


@pytest.mark.extra_large
class EfficientNetV2PresetFullTest(TestCase):
    """
    Test the full enumeration of our preset.
    This every presets for EfficientNetV2 and is only run manually.
    Run with:
    `pytest keras_cv/models/backbones/efficientnet_v2/efficientnet_v2_backbone_presets_test.py --run_extra_large`
    """  # noqa: E501

    @parameterized.named_parameters(
        *[(preset, preset) for preset in EfficientNetV2Backbone.presets]
    )
    def test_load_efficientnet(self, preset):
        input_data = np.ones(shape=(2, 224, 224, 3))
        model = EfficientNetV2Backbone.from_preset(preset)
        model(input_data)

    def test_efficientnet_feature_extractor(self):
        model = EfficientNetV2SBackbone(
            include_rescaling=False,
            input_shape=[256, 256, 3],
        )
        levels = ["P3", "P4"]
        layer_names = [model.pyramid_level_inputs[level] for level in levels]
        backbone_model = get_feature_extractor(model, layer_names, levels)
        inputs = keras.Input(shape=[256, 256, 3])
        outputs = backbone_model(inputs)
        self.assertLen(outputs, 2)
        self.assertEquals(list(outputs.keys()), levels)
        self.assertEquals(outputs["P3"].shape[:3], (None, 32, 32))
        self.assertEquals(outputs["P4"].shape[:3], (None, 16, 16))
