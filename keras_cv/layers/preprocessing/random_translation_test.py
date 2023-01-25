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

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from keras_cv import bounding_box
from keras_cv.layers import preprocessing


class RandomTranslationTest(tf.test.TestCase, parameterized.TestCase):
    def test_random_translation_up_numeric_reflect(self):
        for dtype in (np.int64, np.float32):
            input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1)).astype(dtype)
            # Shifting by -.2 * 5 = 1 pixel.
            layer = preprocessing.RandomTranslation(
                height_factor=(-0.2, -0.2), width_factor=0.0
            )
            output_image = layer(input_image)
            expected_output = np.asarray(
                [
                    [5, 6, 7, 8, 9],
                    [10, 11, 12, 13, 14],
                    [15, 16, 17, 18, 19],
                    [20, 21, 22, 23, 24],
                    [20, 21, 22, 23, 24],
                ]
            ).astype(dtype)
            expected_output = np.reshape(expected_output, (1, 5, 5, 1))
            self.assertAllEqual(expected_output, output_image)

    def test_random_translation_up_numeric_constant(self):
        for dtype in (np.int64, np.float32):
            input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1)).astype(dtype)
            # Shifting by -.2 * 5 = 1 pixel.
            layer = preprocessing.RandomTranslation(
                height_factor=(-0.2, -0.2),
                width_factor=0.0,
                fill_mode="constant",
            )
            output_image = layer(input_image)
            expected_output = np.asarray(
                [
                    [5, 6, 7, 8, 9],
                    [10, 11, 12, 13, 14],
                    [15, 16, 17, 18, 19],
                    [20, 21, 22, 23, 24],
                    [0, 0, 0, 0, 0],
                ]
            ).astype(dtype)
            expected_output = np.reshape(expected_output, (1, 5, 5, 1))
            self.assertAllEqual(expected_output, output_image)

    def test_random_translation_down_numeric_reflect(self):
        for dtype in (np.int64, np.float32):
            input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1)).astype(dtype)
            # Shifting by .2 * 5 = 1 pixel.
            layer = preprocessing.RandomTranslation(
                height_factor=(0.2, 0.2), width_factor=0.0
            )
            output_image = layer(input_image)
            expected_output = np.asarray(
                [
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4],
                    [5, 6, 7, 8, 9],
                    [10, 11, 12, 13, 14],
                    [15, 16, 17, 18, 19],
                ]
            ).astype(dtype)
            expected_output = np.reshape(expected_output, (1, 5, 5, 1))
            self.assertAllEqual(expected_output, output_image)

    def test_random_translation_asymmetric_size_numeric_reflect(self):
        for dtype in (np.int64, np.float32):
            input_image = np.reshape(np.arange(0, 16), (1, 8, 2, 1)).astype(dtype)
            # Shifting by .5 * 8 = 1 pixel.
            layer = preprocessing.RandomTranslation(
                height_factor=(0.5, 0.5), width_factor=0.0
            )
            output_image = layer(input_image)
            # pyformat: disable
            expected_output = np.asarray(
                [
                    [6, 7],
                    [4, 5],
                    [2, 3],
                    [0, 1],
                    [0, 1],
                    [2, 3],
                    [4, 5],
                    [6, 7],
                ]
            ).astype(dtype)
            # pyformat: enable
            expected_output = np.reshape(expected_output, (1, 8, 2, 1))
            self.assertAllEqual(expected_output, output_image)

    def test_random_translation_down_numeric_constant(self):
        for dtype in (np.int64, np.float32):
            input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1)).astype(dtype)
            # Shifting by -.2 * 5 = 1 pixel.
            layer = preprocessing.RandomTranslation(
                height_factor=(0.2, 0.2),
                width_factor=0.0,
                fill_mode="constant",
            )
            output_image = layer(input_image)
            expected_output = np.asarray(
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 2, 3, 4],
                    [5, 6, 7, 8, 9],
                    [10, 11, 12, 13, 14],
                    [15, 16, 17, 18, 19],
                ]
            ).astype(dtype)
            expected_output = np.reshape(expected_output, (1, 5, 5, 1))
            self.assertAllEqual(expected_output, output_image)

    def test_random_translation_left_numeric_reflect(self):
        for dtype in (np.int64, np.float32):
            input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1)).astype(dtype)
            # Shifting by .2 * 5 = 1 pixel.
            layer = preprocessing.RandomTranslation(
                height_factor=0.0, width_factor=(-0.2, -0.2)
            )
            output_image = layer(input_image)
            expected_output = np.asarray(
                [
                    [1, 2, 3, 4, 4],
                    [6, 7, 8, 9, 9],
                    [11, 12, 13, 14, 14],
                    [16, 17, 18, 19, 19],
                    [21, 22, 23, 24, 24],
                ]
            ).astype(dtype)
            expected_output = np.reshape(expected_output, (1, 5, 5, 1))
            self.assertAllEqual(expected_output, output_image)

    def test_random_translation_left_numeric_constant(self):
        for dtype in (np.int64, np.float32):
            input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1)).astype(dtype)
            # Shifting by -.2 * 5 = 1 pixel.
            layer = preprocessing.RandomTranslation(
                height_factor=0.0,
                width_factor=(-0.2, -0.2),
                fill_mode="constant",
            )
            output_image = layer(input_image)
            expected_output = np.asarray(
                [
                    [1, 2, 3, 4, 0],
                    [6, 7, 8, 9, 0],
                    [11, 12, 13, 14, 0],
                    [16, 17, 18, 19, 0],
                    [21, 22, 23, 24, 0],
                ]
            ).astype(dtype)
            expected_output = np.reshape(expected_output, (1, 5, 5, 1))
            self.assertAllEqual(expected_output, output_image)

    def test_random_translation_inference(self):
        input_images = np.random.random((2, 5, 8, 3)).astype(np.float32)
        expected_output = input_images
        layer = preprocessing.RandomTranslation(0.5, 0.5)
        actual_output = layer(input_images, training=False)
        self.assertAllClose(expected_output, actual_output)

    def test_config_with_custom_name(self):
        layer = preprocessing.RandomTranslation(0.5, 0.6, name="image_preproc")
        config = layer.get_config()
        layer_1 = preprocessing.RandomTranslation.from_config(config)
        self.assertEqual(layer_1.name, layer.name)

    def test_unbatched_image(self):
        input_image = np.reshape(np.arange(0, 25), (5, 5, 1)).astype(np.int64)
        # Shifting by -.2 * 5 = 1 pixel.
        layer = preprocessing.RandomTranslation(
            height_factor=(-0.2, -0.2), width_factor=0.0
        )
        output_image = layer(input_image)
        expected_output = np.asarray(
            [
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24],
                [20, 21, 22, 23, 24],
            ]
        ).astype(np.int64)
        expected_output = np.reshape(expected_output, (5, 5, 1))
        self.assertAllEqual(expected_output, output_image)

    def test_output_dtypes(self):
        inputs = np.array([[[1], [2]], [[3], [4]]], dtype="float64")
        layer = preprocessing.RandomTranslation(0.5, 0.6)
        self.assertAllEqual(layer(inputs).dtype, "float32")
        layer = preprocessing.RandomTranslation(0.5, 0.6, dtype="uint8")
        self.assertAllEqual(layer(inputs).dtype, "uint8")
