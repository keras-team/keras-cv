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

import pathlib

import numpy as np
import pytest

from keras_cv.backend import ops
from keras_cv.models.backbones.detectron2.detectron2_aliases import (
    ViTDetBBackbone,
)
from keras_cv.models.backbones.detectron2.detectron2_backbone import (
    ViTDetBackbone,
)
from keras_cv.tests.test_case import TestCase


@pytest.mark.large
class ViTDetPresetSmokeTest(TestCase):
    """
    A smoke test for ViTDet presets we run continuously.
    This only tests the smallest weights we have available. Run with:
    `pytest keras_cv/models/backbones/detectron2/detectron2_backbone_presets_test.py --run_large`  # noqa: E501
    """

    def setUp(self):
        self.input_batch = np.ones(shape=(1, 1024, 1024, 3))

    def test_backbone_output(self):
        model = ViTDetBackbone.from_preset("vitdet_b")
        outputs = model(self.input_batch)

        # The forward pass from a preset should be stable!
        # This test should catch cases where we unintentionally change our
        # network code in a way that would invalidate our preset weights.
        # We should only update these numbers if we are updating a weights
        # file, or have found a discrepancy with the upstream source.

        expected = np.load(
            pathlib.Path(__file__).parent / "data" / "vitdet_b_out.npz"
        )
        # Keep a high tolerance, so we are robust to different hardware.
        self.assertAllClose(
            ops.convert_to_numpy(outputs),
            expected,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_applications_model_output(self):
        model = ViTDetBBackbone()
        model(self.input_batch)

    def test_applications_model_output_with_preset(self):
        model = ViTDetBBackbone.from_preset("vitdet_b")
        model(self.input_batch)

    def test_preset_docstring(self):
        """Check we did our docstring formatting correctly."""
        for name in ViTDetBackbone.presets:
            self.assertRegex(ViTDetBackbone.from_preset.__doc__, name)

    def test_unknown_preset_error(self):
        # Not a preset name
        with self.assertRaises(ValueError):
            ViTDetBackbone.from_preset("vitdet_nonexistant")


@pytest.mark.extra_large
class ViTDetPresetFullTest(TestCase):
    """
    Test the full enumeration of our preset.
    This tests every preset for ViTDet and is only run manually.
    Run with:
    `pytest keras_cv/models/backbones/detectron2/detectron2_backbone_presets_test.py --run_extra_large`  # noqa: E501
    """

    def test_load_ViTDet(self):
        input_data = np.ones(shape=(1, 1024, 1024, 3))
        for preset in ViTDetBackbone.presets:
            model = ViTDetBackbone.from_preset(preset)
            model(input_data)
