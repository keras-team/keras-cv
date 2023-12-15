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

from keras_cv.backend import ops
from keras_cv.layers import preprocessing
from keras_cv.tests.test_case import TestCase


class AutoContrastTest(TestCase):
    def test_constant_channels_dont_get_nanned(self):
        img = np.array([1, 1], dtype=np.float32)
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        layer = preprocessing.AutoContrast(value_range=(0, 255))
        ys = layer(img)

        self.assertTrue(np.any(ops.convert_to_numpy(ys[0]) == 1.0))
        self.assertTrue(np.any(ops.convert_to_numpy(ys[0]) == 1.0))

    def test_auto_contrast_expands_value_range(self):
        img = np.array([0, 128], dtype=np.float32)
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        layer = preprocessing.AutoContrast(value_range=(0, 255))
        ys = layer(img)

        self.assertTrue(np.any(ops.convert_to_numpy(ys[0]) == 0.0))
        self.assertTrue(np.any(ops.convert_to_numpy(ys[0]) == 255.0))

    def test_auto_contrast_different_values_per_channel(self):
        img = np.array(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            dtype=np.float32,
        )
        img = np.expand_dims(img, axis=0)

        layer = preprocessing.AutoContrast(value_range=(0, 255))
        ys = layer(img)

        self.assertTrue(np.any(ops.convert_to_numpy(ys[0, ..., 0]) == 0.0))
        self.assertTrue(np.any(ops.convert_to_numpy(ys[0, ..., 1]) == 0.0))

        self.assertTrue(np.any(ops.convert_to_numpy(ys[0, ..., 0]) == 255.0))
        self.assertTrue(np.any(ops.convert_to_numpy(ys[0, ..., 1]) == 255.0))

        self.assertAllClose(
            ys,
            [
                [
                    [[0.0, 0.0, 0.0], [85.0, 85.0, 85.0]],
                    [[170.0, 170.0, 170.0], [255.0, 255.0, 255.0]],
                ]
            ],
        )

    def test_auto_contrast_expands_value_range_uint8(self):
        img = np.array([0, 128], dtype=np.uint8)
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        layer = preprocessing.AutoContrast(value_range=(0, 255))
        ys = layer(img)

        self.assertTrue(np.any(ops.convert_to_numpy(ys[0]) == 0.0))
        self.assertTrue(np.any(ops.convert_to_numpy(ys[0]) == 255.0))

    def test_auto_contrast_properly_converts_value_range(self):
        img = np.array([0, 0.5], dtype=np.float32)
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        layer = preprocessing.AutoContrast(value_range=(0, 1))
        ys = layer(img)

        self.assertTrue(np.any(ops.convert_to_numpy(ys[0]) == 0.0))
        self.assertTrue(np.any(ops.convert_to_numpy(ys[0]) == 1.0))
