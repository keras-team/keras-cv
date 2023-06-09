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

from keras_cv.models.backbones.densenet.densenet_aliases import (
    DenseNet121Backbone,
)
from keras_cv.models.backbones.densenet.densenet_backbone import (
    DenseNetBackbone,
)


@pytest.mark.large
class DenseNetPresetSmokeTest(tf.test.TestCase, parameterized.TestCase):
    """
    A smoke test for DenseNet presets we run continuously.
    This only tests the smallest weights we have available. Run with:
    `pytest keras_cv/models/backbones/densenet/densenet_backbone_presets_test.py --run_large`  # noqa: E501
    """

    def setUp(self):
        self.input_batch = tf.ones(shape=(2, 224, 224, 3))

    def test_backbone_output(self):
        model = DenseNetBackbone.from_preset("densenet121")
        model(self.input_batch)

    def test_backbone_output_with_weights(self):
        model = DenseNetBackbone.from_preset("densenet121_imagenet")

        # The forward pass from a preset should be stable!
        # This test should catch cases where we unintentionally change our
        # network code in a way that would invalidate our preset weights.
        # We should only update these numbers if we are updating a weights
        # file, or have found a discrepancy with the upstream source.

        outputs = model(tf.ones(shape=(1, 512, 512, 3)))
        expected = [0.0, 0.0, 0.09920305, 0.0, 0.0]
        # Keep a high tolerance, so we are robust to different hardware.
        self.assertAllClose(
            outputs[0, 0, 0, :5], expected, atol=0.01, rtol=0.01
        )

    def test_applications_model_output(self):
        model = DenseNet121Backbone()
        model(self.input_batch)

    def test_applications_model_output_with_preset(self):
        model = DenseNet121Backbone.from_preset("densenet121_imagenet")
        model(self.input_batch)

    def test_preset_docstring(self):
        """Check we did our docstring formatting correctly."""
        for name in DenseNetBackbone.presets:
            self.assertRegex(DenseNetBackbone.from_preset.__doc__, name)

    def test_unknown_preset_error(self):
        # Not a preset name
        with self.assertRaises(ValueError):
            DenseNetBackbone.from_preset("densenet121_clowntown")

    def test_load_weights_error(self):
        # Try to load weights when none available
        with self.assertRaises(ValueError):
            DenseNetBackbone.from_preset("densenet121", load_weights=True)


@pytest.mark.extra_large
class DenseNetPresetFullTest(tf.test.TestCase, parameterized.TestCase):
    """
    Test the full enumeration of our preset.
    This tests every preset for DenseNet and is only run manually.
    Run with:
    `pytest keras_cv/models/backbones/densenet/densenet_backbone_presets_test.py --run_extra_large`  # noqa: E501
    """

    def test_load_densenet(self):
        input_data = tf.ones(shape=(2, 224, 224, 3))
        for preset in DenseNetBackbone.presets:
            model = DenseNetBackbone.from_preset(preset)
            model(input_data)
