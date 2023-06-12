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

"""Tests for loading pretrained model presets."""

import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_cv.models.backbones.mobilenet_v3.mobilenet_v3_backbone import (
    MobileNetV3Backbone,
)


@pytest.mark.large
class MobileNetV3PresetSmokeTest(tf.test.TestCase):
    """
    A smoke test for MobileNetV3 presets we run continuously.
    This only tests the smallest weights we have available. Run with:
    `pytest keras_cv/models/backbones/mobilenet_v3/mobilenet_v3_backbone_presets_test.py --run_large`
    """  # noqa: E501

    def setUp(self):
        self.input_batch = tf.ones(shape=(8, 224, 224, 3))

    def test_backbone_output(self):
        model = MobileNetV3Backbone.from_preset("mobilenet_v3_large_imagenet")
        outputs = model(self.input_batch)

        # The forward pass from a preset should be stable!
        # This test should catch cases where we unintentionally change our
        # network code in a way that would invalidate our preset weights.
        # We should only update these numbers if we are updating a weights
        # file, or have found a discrepancy with the upstream source.
        outputs = outputs[0, 0, 0, :5]
        expected = [0.27, 0.01, 0.29, 0.08, -0.12]
        # Keep a high tolerance, so we are robust to different hardware.
        self.assertAllClose(outputs, expected, atol=0.01, rtol=0.01)


@pytest.mark.extra_large
class MobileNetV3PresetFullTest(tf.test.TestCase, parameterized.TestCase):
    """
    Test the full enumeration of our preset.
    This tests every preset for MobileNetV3 and is only run manually.
    Run with:
    `pytest keras_cv/models/backbones/mobilenet_v3/mobilenet_v3_backbone_presets_test.py --run_extra_large`
    """  # noqa: E501

    def test_load_mobilenet_v3(self):
        input_data = tf.ones(shape=(2, 224, 224, 3))
        for preset in MobileNetV3Backbone.presets:
            model = MobileNetV3Backbone.from_preset(preset)
            model(input_data)
