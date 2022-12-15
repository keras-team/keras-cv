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
import unittest

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from keras_cv.layers.preprocessing.random_crop import RandomCrop


class RandomCropTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("random_crop_4_by_6", 4, 6),
        ("random_crop_3_by_2", 3, 2),
        ("random_crop_full_height", 5, 2),
        ("random_crop_full_width", 3, 8),
    )
    def test_output_shape(self, expected_height, expected_width):
        np.random.seed(1337)
        num_samples = 2
        orig_height = 5
        orig_width = 8
        channels = 3
        input = tf.random.uniform(
            shape=[num_samples, orig_height, orig_width, channels],
        )
        layer = RandomCrop(expected_height, expected_width)
        actual_output = layer(input)
        expected_output = tf.random.uniform(
            shape=(
                num_samples,
                expected_height,
                expected_width,
                channels,
            ),
        )
        self.assertAllEqual(expected_output.shape, actual_output.shape)

    def test_input_smaller_than_crop_box(self):
        np.random.seed(1337)
        height, width = 10, 8
        inp = np.random.random((12, 3, 3, 3))
        layer = RandomCrop(height, width)
        actual_output = layer(inp)
        # In this case, output should equal resizing with crop_to_aspect
        # ratio.
        resizing_layer = tf.keras.layers.Resizing(height, width)
        expected_output = resizing_layer(inp)
        self.assertAllEqual(expected_output, actual_output)

    def test_training_with_mock(self):
        np.random.seed(1337)
        height, width = 3, 4
        height_offset = np.random.randint(low=0, high=3)
        width_offset = np.random.randint(low=0, high=5)
        mock_offset = [height_offset, width_offset]
        layer = RandomCrop(height, width)
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            return_value=mock_offset,
        ):
            inp = np.random.random((12, 5, 8, 3))
            actual_output = layer(inp, training=True)
            expected_output = inp[
                :,
                height_offset : (height_offset + height),
                width_offset : (width_offset + width),
                :,
            ]
            self.assertAllClose(expected_output, actual_output)

    def test_random_crop_full(self):
        np.random.seed(1337)
        height, width = 8, 16
        inp = np.random.random((12, 8, 16, 3))
        layer = RandomCrop(height, width)
        actual_output = layer(inp, training=False)
        self.assertAllClose(inp, actual_output)

    def test_config_with_custom_name(self):
        layer = RandomCrop(5, 5, name="image_preproc")
        config = layer.get_config()
        layer_1 = RandomCrop.from_config(config)
        self.assertEqual(layer_1.name, layer.name)

    def test_unbatched_image(self):
        np.random.seed(1337)
        inp = np.random.random((16, 16, 3))
        mock_offset = [2, 2]
        layer = RandomCrop(8, 8)
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            return_value=mock_offset,
        ):
            actual_output = layer(inp, training=True)
            self.assertAllClose(inp[2:10, 2:10, :], actual_output)

    def test_batched_input(self):
        np.random.seed(1337)
        inp = np.random.random((20, 16, 16, 3))
        mock_offset = [2, 2]
        layer = RandomCrop(8, 8)
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            return_value=mock_offset,
        ):
            actual_output = layer(inp, training=True)
            self.assertAllClose(inp[:, 2:10, 2:10, :], actual_output)

    def test_output_dtypes(self):
        inputs = np.array([[[1], [2]], [[3], [4]]], dtype="float64")
        layer = RandomCrop(2, 2)
        self.assertAllEqual(layer(inputs).dtype, "float32")
        layer = RandomCrop(2, 2, dtype="uint8")
        self.assertAllEqual(layer(inputs).dtype, "uint8")

    def test_compute_output_signature(self):
        inputs = np.random.random((2, 16, 16, 3))
        layer = RandomCrop(2, 2)
        output = layer(inputs)
        tf.print(output.shape)
        output_signature = layer.compute_image_signature(inputs).shape
        tf.print(output_signature)
        self.assertAllEqual(output.shape, output_signature)
