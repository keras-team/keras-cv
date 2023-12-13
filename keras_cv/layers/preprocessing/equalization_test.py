# Copyright 2022 The KerasCV Authors
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

import numpy as np
import pytest
from absl.testing import parameterized

from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.layers.preprocessing.equalization import Equalization
from keras_cv.tests.test_case import TestCase


class EqualizationTest(TestCase):
    def test_return_shapes(self):
        xs = 255 * np.ones((2, 512, 512, 3), dtype=np.int32)
        layer = Equalization(value_range=(0, 255))
        xs = layer(xs)

        self.assertEqual(xs.shape, (2, 512, 512, 3))
        self.assertAllEqual(xs, 255 * np.ones((2, 512, 512, 3)))

    @pytest.mark.tf_keras_only
    def test_return_shapes_inside_model(self):
        layer = Equalization(value_range=(0, 255))
        inp = keras.layers.Input(shape=[512, 512, 5])
        out = layer(inp)
        model = keras.models.Model(inp, out)

        self.assertEqual(model.output_shape, (None, 512, 512, 5))

    def test_equalizes_to_all_bins(self):
        xs = np.random.uniform(size=(2, 512, 512, 3), low=0, high=255).astype(
            np.float32
        )
        layer = Equalization(value_range=(0, 255))
        xs = layer(xs)

        for i in range(0, 256):
            self.assertTrue(np.any(ops.convert_to_numpy(xs) == i))

    @parameterized.named_parameters(
        ("float32", np.float32), ("int32", np.int32), ("int64", np.int64)
    )
    def test_input_dtypes(self, dtype):
        xs = np.random.uniform(size=(2, 512, 512, 3), low=0, high=255).astype(
            dtype
        )
        layer = Equalization(value_range=(0, 255))
        xs = ops.convert_to_numpy(layer(xs))

        for i in range(0, 256):
            self.assertTrue(np.any(xs == i))
        self.assertAllInRange(xs, 0, 255)

    @parameterized.named_parameters(("0_255", 0, 255), ("0_1", 0, 1))
    def test_output_range(self, lower, upper):
        xs = np.random.uniform(
            size=(2, 512, 512, 3), low=lower, high=upper
        ).astype(np.float32)
        layer = Equalization(value_range=(lower, upper))
        xs = ops.convert_to_numpy(layer(xs))
        self.assertAllInRange(xs, lower, upper)
