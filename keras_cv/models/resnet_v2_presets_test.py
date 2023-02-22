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

from keras_cv.models.resnet_v2_backbone import ResNetV2Backbone


@pytest.mark.large
class ResNetV2PresetSmokeTest(tf.test.TestCase, parameterized.TestCase):
    """
    A smoke test for ResNetV2 presets we run continuously.
    This only tests the smallest weights we have available. Run with:
    `pytest keras_cv/models//resnetv2_presets_test.py --run_large`
    """

    @parameterized.named_parameters(
        ("load_weights", True), ("no_load_weights", False)
    )
    def test_backbone_output(self, load_weights):
        input_data = tf.ones(shape=(8, 224, 224, 3))
        model = ResNetV2Backbone.from_preset(
            "resnet50_v2", load_weights=load_weights
        )
        outputs = model(input_data)

        if load_weights:
            # The forward pass from a preset should be stable!
            # This test should catch cases where we unintentionally change our
            # network code in a way that would invalidate our preset weights.
            # We should only update these numbers if we are updating a weights
            # file, or have found a discrepancy with the upstream source.
            outputs = outputs[0, 0, 0, :5]
            expected = [1.051145, 0, 0, 1.16328, 0]
            # Keep a high tolerance, so we are robust to different hardware.
            self.assertAllClose(outputs, expected, atol=0.01, rtol=0.01)

    @parameterized.named_parameters(
        ("backbone", ResNetV2Backbone),
    )
    def test_preset_docstring(self, cls):
        """Check we did our docstring formatting correctly."""
        for name in cls.presets:
            self.assertRegex(cls.from_preset.__doc__, name)

    @parameterized.named_parameters(
        ("backbone", ResNetV2Backbone),
    )
    def test_unknown_preset_error(self, cls):
        # Not a preset name
        with self.assertRaises(ValueError):
            cls.from_preset("resnet_v2_clowntown")


@pytest.mark.extra_large
class ResNetV2PresetFullTest(tf.test.TestCase, parameterized.TestCase):
    """
    Test the full enumeration of our preset.
    This every presets for ResNetV2 and is only run manually.
    Run with:
    `pytest keras_cv/models/resnet_v2_presets_test.py --run_extra_large`
    """

    @parameterized.named_parameters(
        ("load_weights", True), ("no_load_weights", False)
    )
    def test_load_resnetv2(self, load_weights):
        input_data = tf.ones(shape=(8, 224, 224, 3))
        for preset in ResNetV2Backbone.presets:
            try:
                model = ResNetV2Backbone.from_preset(
                    preset, load_weights=load_weights
                )
                model(input_data)
            except ValueError as err:
                # Only allow "no weights available" error
                if (
                    load_weights
                    and str(err).find("Pretrained weights not available") < 0
                ):
                    raise ValueError(err)
