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

from keras_cv.models.backbones.efficientnetv2.efficientnetv2_backbone import (
    EfficientNetV2Backbone,
)
from keras_cv.models.backbones.efficientnetv2.efficientnetv2_backbone import (
    EfficientNetV2MBackbone,
)
from keras_cv.utils.train import get_feature_extractor


@pytest.mark.large
class EfficientNetV2PresetSmokeTest(tf.test.TestCase, parameterized.TestCase):
    """
    A smoke test for EfficientNetV2 presets we run continuously.
    This only tests the smallest weights we have available. Run with:
    `pytest keras_cv/models/backbones/efficientnetv2/efficientnetv2_backbone_presets_test.py --run_large` # noqa: E501
    """

    def setUp(self):
        self.input_batch = tf.ones(shape=(2, 224, 224, 3))

    @parameterized.named_parameters(
        ("preset_with_weights", "efficientnetv250_imagenet"),
        # ("preset_no_weights", "efficientnetv250"),
    )
    def test_backbone_output(self):
        model = EfficientNetV2Backbone.from_preset("efficientnetv250")
        model(self.input_batch)

    def test_backbone_output_with_weights(self):
        model = EfficientNetV2Backbone.from_preset("efficientnetv250_imagenet")

        # initialize trainable networks
        extractor_levels = [3, 4, 5]
        extractor_layer_names = [
            model.pyramid_level_inputs[i] for i in extractor_levels
        ]
        feature_extractor = get_feature_extractor(
            model, extractor_layer_names, extractor_levels
        )

        # The forward pass from a preset should be stable!
        # This test should catch cases where we unintentionally change our
        # network code in a way that would invalidate our preset weights.
        # We should only update these numbers if we are updating a weights
        # file, or have found a discrepancy with the upstream source.

        outputs = feature_extractor(tf.ones(1, 512, 512, 3))
        outputs = [outputs[i][0, 0, 0, :5] for i in extractor_levels]
        expected = [
            [0.50249904, 0.35751498, 0.9474212, 1.0659311, 1.1105202],
            [0.65718395, 0.7209194, 0.39707005, 0.5164382, 0.73338735],
            [
                1.0615100e00,
                1.2732037e00,
                4.2707797e-02,
                6.3376129e-04,
                1.0917966e00,
            ],
            [0.0, 0.0, 0.0, 0.05175382, 0.0],
        ]
        # Keep a high tolerance, so we are robust to different hardware.
        self.assertAllClose(outputs, expected, atol=0.01, rtol=0.01)

    def test_applications_model_output(self):
        model = EfficientNetV2MBackbone()
        model(self.input_batch)

    def test_applications_model_output_with_preset(self):
        model = EfficientNetV2MBackbone.from_preset("efficientnetv250_imagenet")
        model(self.input_batch)

    def test_preset_docstring(self):
        """Check we did our docstring formatting correctly."""
        for name in EfficientNetV2Backbone.presets:
            self.assertRegex(EfficientNetV2Backbone.from_preset.__doc__, name)

    def test_unknown_preset_error(self):
        # Not a preset name
        with self.assertRaises(ValueError):
            EfficientNetV2Backbone.from_preset("efficientnetv250_clowntown")

    def test_load_weights_error(self):
        # Try to load weights when none available
        with self.assertRaises(ValueError):
            EfficientNetV2Backbone.from_preset(
                "efficientnetv250", load_weights=True
            )


@pytest.mark.extra_large
class EfficientNetV2PresetFullTest(tf.test.TestCase, parameterized.TestCase):
    """
    Test the full enumeration of our preset.
    This every presets for EfficientNetV2 and is only run manually.
    Run with:
    `pytest keras_cv/models/backbones/efficientnetv2_v1/efficientnetv2_backbone_presets_test.py --run_extra_large`  # noqa: E501
    """

    def test_load_efficientnetv2(self):
        input_data = tf.ones(shape=(2, 224, 224, 3))
        for preset in EfficientNetV2Backbone.presets:
            model = EfficientNetV2Backbone.from_preset(preset)
            model(input_data)
