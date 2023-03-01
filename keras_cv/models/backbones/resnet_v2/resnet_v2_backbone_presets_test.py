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

from keras_cv.models.backbones.resnet_v2.resnet_v2_backbone import (
    ResNet50V2Backbone,
)
from keras_cv.models.backbones.resnet_v2.resnet_v2_backbone import (
    ResNetV2Backbone,
)


@pytest.mark.large
class ResNetV2PresetSmokeTest(tf.test.TestCase, parameterized.TestCase):
    """
    A smoke test for ResNetV2 presets we run continuously.
    This only tests the smallest weights we have available. Run with:
    `pytest keras_cv/models/backbones/resnet_v2/resnetv2_presets_test.py --run_large`
    """

    def setUp(self):
        self.input_batch = tf.ones(shape=(8, 224, 224, 3))

    @parameterized.named_parameters(
        ("preset_with_weights", "resnet50_v2_imagenet"),
        ("preset_no_weights", "resnet50_v2"),
    )
    def test_backbone_output(self, preset):
        model = ResNetV2Backbone.from_preset(preset)
        outputs = model(self.input_batch)

        if preset == "resnet50_v2_imagenet":
            # The forward pass from a preset should be stable!
            # This test should catch cases where we unintentionally change our
            # network code in a way that would invalidate our preset weights.
            # We should only update these numbers if we are updating a weights
            # file, or have found a discrepancy with the upstream source.
            outputs = outputs[0, 0, 0, :5]
            expected = [1.051145, 0, 0, 1.16328, 0]
            # Keep a high tolerance, so we are robust to different hardware.
            self.assertAllClose(outputs, expected, atol=0.01, rtol=0.01)

    def test_applications_model_output(self):
        model = ResNet50V2Backbone()
        model(self.input_batch)

    def test_applications_model_output_with_preset(self):
        model = ResNet50V2Backbone.from_preset("resnet50_v2_imagenet")
        model(self.input_batch)

    def test_preset_docstring(self):
        """Check we did our docstring formatting correctly."""
        for name in ResNetV2Backbone.presets:
            self.assertRegex(ResNetV2Backbone.from_preset.__doc__, name)

    def test_unknown_preset_error(self):
        # Not a preset name
        with self.assertRaises(ValueError):
            ResNetV2Backbone.from_preset("resnet50_v2_clowntown")

    def test_load_weights_error(self):
        # Try to load weights when none available
        with self.assertRaises(ValueError):
            ResNetV2Backbone.from_preset("resnet50_v2", load_weights=True)


@pytest.mark.extra_large
class ResNetV2PresetFullTest(tf.test.TestCase, parameterized.TestCase):
    """
    Test the full enumeration of our preset.
    This every presets for ResNetV2 and is only run manually.
    Run with:
    `pytest keras_cv/models/backbones/resnet_v2/resnet_v2_presets_test.py --run_extra_large`
    """

    def test_load_resnetv2(self):
        input_data = tf.ones(shape=(8, 224, 224, 3))
        for preset in ResNetV2Backbone.presets:
            model = ResNetV2Backbone.from_preset(preset)
            model(input_data)
