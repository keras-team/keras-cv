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

from keras_cv.models.backbones.mobilenet_v3.mobilenet_v3_backbone import (
    MobileNetV3Backbone,
)


@pytest.mark.extra_large
class MobileNetV3PresetFullTest(tf.test.TestCase, parameterized.TestCase):
    """
    Test the full enumeration of our preset.
    This tests every preset for MobileNetV3 and is only run manually.
    Run with:
    `pytest keras_cv/models/backbones/mobilenet_v3/mobilenet_v3_backbone_presets_test.py --run_extra_large`
    """

    def test_load_resnet(self):
        input_data = tf.ones(shape=(2, 224, 224, 3))
        for preset in MobileNetV3Backbone.presets:
            model = MobileNetV3Backbone.from_preset(preset)
            model(input_data)
