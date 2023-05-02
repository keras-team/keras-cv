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

from keras_cv.models import YOLOV8Backbone
from keras_cv.models.backbones.csp_darknet import csp_darknet_backbone
from keras_cv.models.backbones.csp_darknet import legacy
from keras_cv.models.object_detection.yolo_v8.yolo_v8_backbone_presets import (
    copy_weights,
)


@pytest.mark.large
class CSPDarkNetPresetSmokeTest(tf.test.TestCase, parameterized.TestCase):
    """
    A smoke test for CSPDarkNet presets we run continuously.
    This only tests the smallest weights we have available. Run with:
    `pytest keras_cv/models/backbones/csp_darknet/csp_darknet_backbone_presets_test.py --run_large`  # noqa: E501
    """

    def setUp(self):
        self.input_batch = tf.ones(shape=(2, 224, 224, 3))

    def test_backbone_output(self):
        model = csp_darknet_backbone.CSPDarkNetBackbone.from_preset(
            "csp_darknet_tiny"
        )
        model(self.input_batch)

    def test_backbone_output_with_weights_tiny(self):
        model = csp_darknet_backbone.CSPDarkNetBackbone.from_preset(
            "csp_darknet_tiny_imagenet"
        )
        outputs = model(tf.ones(shape=(1, 512, 512, 3)))

        # The forward pass from a preset should be stable!
        # This test should catch cases where we unintentionally change our
        # network code in a way that would invalidate our preset weights.
        # We should only update these numbers if we are updating a weights
        # file, or have found a discrepancy with the upstream source.

        expected = [-0.16216235, 0.7333651, 0.4312072, 0.738807, -0.2515305]
        # Keep a high tolerance, so we are robust to different hardware.
        self.assertAllClose(
            outputs[0, 0, 0, :5], expected, atol=0.01, rtol=0.01
        )

    def test_applications_model_output(self):
        model = csp_darknet_backbone.CSPDarkNetMBackbone()
        model(self.input_batch)

    def test_applications_model_output_with_preset(self):
        model = csp_darknet_backbone.CSPDarkNetBackbone.from_preset(
            "csp_darknet_tiny_imagenet"
        )
        model(self.input_batch)

    def test_preset_docstring(self):
        """Check we did our docstring formatting correctly."""
        for name in csp_darknet_backbone.CSPDarkNetBackbone.presets:
            self.assertRegex(
                csp_darknet_backbone.CSPDarkNetBackbone.from_preset.__doc__,
                name,
            )

    def test_unknown_preset_error(self):
        # Not a preset name
        with self.assertRaises(ValueError):
            csp_darknet_backbone.CSPDarkNetBackbone.from_preset(
                "unknown_weights"
            )

    def test_load_weights_error(self):
        # Try to load weights when none available
        with self.assertRaises(ValueError):
            csp_darknet_backbone.CSPDarkNetBackbone.from_preset(
                "csp_darknet_tiny", load_weights=True
            )

    @parameterized.named_parameters(
        ("xs", "yolov8_xs_backbone_coco", "yolov8_xs_backbone"),
        ("s", "yolov8_s_backbone_coco", "yolov8_s_backbone"),
        ("m", "yolov8_m_backbone_coco", "yolov8_m_backbone"),
        ("l", "yolov8_l_backbone_coco", "yolov8_l_backbone"),
        ("xl", "yolov8_xl_backbone_coco", "yolov8_xl_backbone"),
    )
    def test_yolo_v8_preset_same_output(self, yolo_preset, csp_preset):
        yolo_model = YOLOV8Backbone.from_preset(yolo_preset)
        csp_model = csp_darknet_backbone.CSPDarkNetBackbone.from_preset(
            csp_preset
        )
        copy_weights(yolo_model, csp_model)
        outputs = csp_model(tf.ones(shape=(1, 512, 512, 3)))
        expected = yolo_model(tf.ones(shape=(1, 512, 512, 3)))
        self.assertAllClose(outputs, expected)

        csp_model = csp_darknet_backbone.CSPDarkNetBackbone.from_preset(
            yolo_preset
        )
        outputs = csp_model(tf.ones(shape=(1, 512, 512, 3)))
        self.assertAllClose(outputs, expected)

    @parameterized.named_parameters(
        ("tiny", "csp_darknet_tiny_imagenet", "csp_darknet_tiny"),
        ("l", "csp_darknet_l_imagenet", "csp_darknet_l"),
    )
    def test_legacy_csp_preset_same_output(self, old_csp_preset, csp_preset):
        old_csp_model = (
            legacy.csp_darknet_backbone.CSPDarkNetBackbone.from_preset(
                old_csp_preset
            )
        )
        csp_model = csp_darknet_backbone.CSPDarkNetBackbone.from_preset(
            csp_preset
        )
        copy_weights(old_csp_model, csp_model)
        outputs = csp_model(tf.ones(shape=(1, 512, 512, 3)))
        expected = old_csp_model(tf.ones(shape=(1, 512, 512, 3)))
        self.assertAllClose(outputs, expected)

        csp_model = csp_darknet_backbone.CSPDarkNetBackbone.from_preset(
            old_csp_preset
        )
        outputs = csp_model(tf.ones(shape=(1, 512, 512, 3)))
        self.assertAllClose(outputs, expected)


@pytest.mark.extra_large
class CSPDarkNetPresetFullTest(tf.test.TestCase, parameterized.TestCase):
    """
    Test the full enumeration of our preset.
    This every presets for CSPDarkNet and is only run manually.
    Run with:
    `pytest keras_cv/models/backbones/csp_darknet/csp_darknet_backbone_presets_test.py --run_extra_large`  # noqa: E501
    """

    def test_load_csp_darknet(self):
        input_data = tf.ones(shape=(2, 512, 512, 3))
        for preset in csp_darknet_backbone.CSPDarkNetBackbone.presets:
            model = csp_darknet_backbone.CSPDarkNetBackbone.from_preset(preset)
            model(input_data)
