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

from keras_cv.models.backbones.vit.vit_backbone import ViTBackbone


@pytest.mark.large
class ViTPresetSmokeTest(tf.test.TestCase, parameterized.TestCase):
    """
    A smoke test for ViT to test continuously.
    Only tests the smallest available weights, i.e. ViT-Tiny16. Run with:
    `pytest keras_cv/models/backbones/vit/vit_backbone_presets_test.py
    --run_large`
    """

    def setUp(self):
        self.input_batch = tf.ones(shape=(2, 224, 224, 3))

    @parameterized.named_parameters(("preset_with_weights", "vittiny16"))
    def test_backbone_output(self):
        model = ViTBackbone.from_preset("vittiny16")
        model(self.input_batch)

    def test_backbone_output_with_weights(self):
        model = ViTBackbone.from_preset("vittiny16_imagenet")
        outputs = model(tf.ones(shape=(1, 224, 224, 3)))

        test_output = outputs[0, :5, :5]
        expected_output = [
            [6.897311, 0.01724231, 0.58274484, -0.8644532, 2.0606039],
            [4.713185, 0.9128607, 0.60258734, 1.8128879, -0.10353164],
            [1.8353163, -2.883653, 2.8262897, 3.3035126, 1.267294],
            [-0.03964591, -1.74914, 1.0305521, 2.1088347, 1.366538],
            [-0.669904, -2.6013992, 1.5256371, 2.7898643, 1.2124418],
        ]

        self.assertAllClose(test_output, expected_output, rtol=0.01, atol=0.01)
