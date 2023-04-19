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


import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_backbone import (
    EfficientNetV2Backbone,
)


@pytest.mark.extra_large
class EfficientNetV2PresetFullTest(tf.test.TestCase, parameterized.TestCase):
    """
    Test the full enumeration of our preset.
    This every presets for EfficientNetV2 and is only run manually.
    Run with:
    `pytest keras_cv/models/backbones/efficientnet_v2/efficientnet_v2_backbone_presets_test.py --run_extra_large`
    """

    @parameterized.named_parameters(
        *[(preset, preset) for preset in EfficientNetV2Backbone.presets]
    )
    def test_load_efficientnet(self, preset):
        input_data = tf.ones(shape=(2, 224, 224, 3))
        model = EfficientNetV2Backbone.from_preset(preset)
        model(input_data)
