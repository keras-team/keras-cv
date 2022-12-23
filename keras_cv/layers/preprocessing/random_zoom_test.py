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
import tensorflow as tf
from absl.testing import parameterized

from keras_cv.layers.preprocessing.random_zoom import RandomZoom


class RandomZoomTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("random_zoom_in_4_by_6", -0.4, -0.6),
        ("random_zoom_in_2_by_3", -0.2, -0.3),
        ("random_zoom_in_tuple_factor", (-0.4, -0.5), (-0.2, -0.3)),
        ("random_zoom_out_4_by_6", 0.4, 0.6),
        ("random_zoom_out_2_by_3", 0.2, 0.3),
        ("random_zoom_out_tuple_factor", (0.4, 0.5), (0.2, 0.3)),
    )
    def test_output_shapes(self, height_factor, width_factor):
        np.random.seed(1337)
        num_samples = 2
        orig_height = 5
        orig_width = 8
        channels = 3
        input = tf.random.uniform(
            shape=[num_samples, orig_height, orig_width, channels],
        )
        layer = RandomZoom(height_factor, width_factor)
        actual_output = layer(input)
        expected_output = tf.random.uniform(
            shape=(
                num_samples,
                orig_height,
                orig_width,
                channels,
            ),
        )
        self.assertAllEqual(expected_output.shape, actual_output.shape)

    def test_random_zoom_in_numeric(self):
        for dtype in (np.int64, np.float32):
            input_image = np.reshape(np.arange(0, 25), (5, 5, 1)).astype(dtype)
            layer = RandomZoom((-0.5, -0.5), (-0.5, -0.5), interpolation="nearest")
            output_image = layer(np.expand_dims(input_image, axis=0))
            expected_output = np.asarray(
                [
                    [6, 7, 7, 8, 8],
                    [11, 12, 12, 13, 13],
                    [11, 12, 12, 13, 13],
                    [16, 17, 17, 18, 18],
                    [16, 17, 17, 18, 18],
                ]
            ).astype(dtype)
            expected_output = np.reshape(expected_output, (1, 5, 5, 1))
            self.assertAllEqual(expected_output, output_image)

    def test_random_zoom_out_numeric(self):
        for dtype in (np.int64, np.float32):
            input_image = np.reshape(np.arange(0, 25), (5, 5, 1)).astype(dtype)
            layer = RandomZoom(
                (0.5, 0.5),
                (0.8, 0.8),
                fill_mode="constant",
                interpolation="nearest",
            )
            output_image = layer(np.expand_dims(input_image, axis=0))
            expected_output = np.asarray(
                [
                    [0, 0, 0, 0, 0],
                    [0, 5, 7, 9, 0],
                    [0, 10, 12, 14, 0],
                    [0, 20, 22, 24, 0],
                    [0, 0, 0, 0, 0],
                ]
            ).astype(dtype)
            expected_output = np.reshape(expected_output, (1, 5, 5, 1))
            self.assertAllEqual(expected_output, output_image)

    def test_random_zoom_out_numeric_preserve_aspect_ratio(self):
        for dtype in (np.int64, np.float32):
            input_image = np.reshape(np.arange(0, 25), (5, 5, 1)).astype(dtype)
            layer = RandomZoom(
                (0.5, 0.5), fill_mode="constant", interpolation="nearest"
            )
            output_image = layer(np.expand_dims(input_image, axis=0))
            expected_output = np.asarray(
                [
                    [0, 0, 0, 0, 0],
                    [0, 6, 7, 9, 0],
                    [0, 11, 12, 14, 0],
                    [0, 21, 22, 24, 0],
                    [0, 0, 0, 0, 0],
                ]
            ).astype(dtype)
            expected_output = np.reshape(expected_output, (1, 5, 5, 1))
            self.assertAllEqual(expected_output, output_image)

    def test_random_zoom_inference(self):
        input_images = np.random.random((2, 5, 8, 3)).astype(np.float32)
        expected_output = input_images
        layer = RandomZoom(0.5, 0.5)
        actual_output = layer(input_images, training=False)
        self.assertAllClose(expected_output, actual_output)

    def test_config_with_custom_name(self):
        layer = RandomZoom(0.5, 0.6, name="image_preproc")
        config = layer.get_config()
        layer_1 = RandomZoom.from_config(config)
        self.assertEqual(layer_1.name, layer.name)

    def test_unbatched_image(self):
        input_image = np.reshape(np.arange(0, 25), (5, 5, 1)).astype(np.int64)
        layer = RandomZoom((-0.5, -0.5), (-0.5, -0.5), interpolation="nearest")
        output_image = layer(input_image)
        expected_output = np.asarray(
            [
                [6, 7, 7, 8, 8],
                [11, 12, 12, 13, 13],
                [11, 12, 12, 13, 13],
                [16, 17, 17, 18, 18],
                [16, 17, 17, 18, 18],
            ]
        ).astype(np.int64)
        expected_output = np.reshape(expected_output, (5, 5, 1))
        self.assertAllEqual(expected_output, output_image)

    def test_output_dtypes(self):
        inputs = np.array([[[1], [2]], [[3], [4]]], dtype="float64")
        layer = RandomZoom(0.5, 0.5)
        self.assertAllEqual(layer(inputs).dtype, "float32")
        layer = RandomZoom(0.5, 0.5, dtype="uint8")
        self.assertAllEqual(layer(inputs).dtype, "uint8")
