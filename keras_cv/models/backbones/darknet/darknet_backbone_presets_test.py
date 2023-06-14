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

from keras_cv.models.backbones.darknet.darknet_backbone import DarkNet53Backbone
from keras_cv.models.backbones.darknet.darknet_backbone import DarkNetBackbone


@pytest.mark.large
class DarkNetPresetSmokeTest(tf.test.TestCase, parameterized.TestCase):
    """
    A smoke test for DarkNet presets we run continuously.
    This only tests the smallest weights we have available. Run with:
    `pytest keras_cv/models/backbones/darknet/darknet_backbone_presets_test.py --run_large`  # noqa: E501
    """

    def setUp(self):
        self.input_batch = tf.ones(shape=(2, 224, 224, 3))

    def test_backbone_output(self):
        model = DarkNetBackbone.from_preset("darknet53")
        model(self.input_batch)

    def test_backbone_output_with_weights(self):
        model = DarkNetBackbone.from_preset("darknet53_imagenet")

        # The forward pass from a preset should be stable!
        # This test should catch cases where we unintentionally change our
        # network code in a way that would invalidate our preset weights.
        # We should only update these numbers if we are updating a weights
        # file, or have found a discrepancy with the upstream source.

        outputs = model(tf.ones(shape=(1, 512, 512, 3)))
        expected = [-0.04739833, 2.6341133, -0.03298496, 1.7416457, 0.10866892]
        # Keep a high tolerance, so we are robust to different hardware.
        self.assertAllClose(
            outputs[0, 0, 0, :5], expected, atol=0.01, rtol=0.01
        )

    def test_applications_model_output(self):
        model = DarkNet53Backbone()
        model(self.input_batch)

    def test_applications_model_output_with_preset(self):
        model = DarkNet53Backbone.from_preset("darknet53_imagenet")
        model(self.input_batch)

    def test_preset_docstring(self):
        """Check we did our docstring formatting correctly."""
        for name in DarkNetBackbone.presets:
            self.assertRegex(DarkNetBackbone.from_preset.__doc__, name)

    def test_unknown_preset_error(self):
        # Not a preset name
        with self.assertRaises(ValueError):
            DarkNetBackbone.from_preset("darknet53_clowntown")

    def test_load_weights_error(self):
        # Try to load weights when none available
        with self.assertRaises(ValueError):
            DarkNetBackbone.from_preset("darknet53", load_weights=True)


@pytest.mark.extra_large
class DarkNetPresetFullTest(tf.test.TestCase, parameterized.TestCase):
    """
    Test the full enumeration of our preset.
    This tests every preset for DarkNet and is only run manually.
    Run with:
    `pytest keras_cv/models/backbones/darknet/darknet_backbone_presets_test.py --run_extra_large`  # noqa: E501
    """

    def test_load_darknet(self):
        input_data = tf.ones(shape=(2, 224, 224, 3))
        for preset in DarkNetBackbone.presets:
            model = DarkNetBackbone.from_preset(preset)
            model(input_data)
