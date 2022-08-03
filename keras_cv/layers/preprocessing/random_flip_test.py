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

from keras_cv.layers.preprocessing.random_flip import RandomFlip


class RandomFlipTest(tf.test.TestCase, parameterized.TestCase):
    def test_horizontal_flip(self):
        np.random.seed(1337)
        mock_random = [0.6, 0.6]
        inp = np.random.random((2, 5, 8, 3))
        expected_output = np.flip(inp, axis=2)
        layer = RandomFlip("horizontal")
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            side_effect=mock_random,
        ):
            actual_output = layer(inp, training=True)
            self.assertAllClose(expected_output, actual_output)

    def test_vertical_flip(self):
        np.random.seed(1337)
        mock_random = [0.6, 0.6]
        inp = np.random.random((2, 5, 8, 3))
        expected_output = np.flip(inp, axis=1)
        layer = RandomFlip("vertical")
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            side_effect=mock_random,
        ):
            actual_output = layer(inp, training=True)
            self.assertAllClose(expected_output, actual_output)

    def test_flip_both(self):
        np.random.seed(1337)
        mock_random = [0.6, 0.6, 0.6, 0.6]
        inp = np.random.random((2, 5, 8, 3))
        expected_output = np.flip(inp, axis=2)
        expected_output = np.flip(expected_output, axis=1)
        layer = RandomFlip("horizontal_and_vertical")
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            side_effect=mock_random,
        ):
            actual_output = layer(inp, training=True)
            self.assertAllClose(expected_output, actual_output)

    def test_random_flip_inference(self):
        input_images = np.random.random((2, 5, 8, 3)).astype(np.float32)
        expected_output = input_images
        layer = RandomFlip()
        actual_output = layer(input_images, training=False)
        self.assertAllClose(expected_output, actual_output)

    def test_random_flip_default(self):
        input_images = np.random.random((2, 5, 8, 3)).astype(np.float32)
        expected_output = np.flip(np.flip(input_images, axis=1), axis=2)
        mock_random = [0.6, 0.6, 0.6, 0.6]
        layer = RandomFlip()
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            side_effect=mock_random,
        ):
            actual_output = layer(input_images, training=True)
            self.assertAllClose(expected_output, actual_output)

    def test_config_with_custom_name(self):
        layer = RandomFlip(name="image_preproc")
        config = layer.get_config()
        layer_1 = RandomFlip.from_config(config)
        self.assertEqual(layer_1.name, layer.name)

    def test_random_flip_unbatched_image(self):
        input_image = np.random.random((4, 4, 1)).astype(np.float32)
        expected_output = np.flip(input_image, axis=0)
        mock_random = [0.6, 0.6, 0.6, 0.6]
        layer = RandomFlip("vertical")
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            side_effect=mock_random,
        ):
            actual_output = layer(input_image, training=True)
            self.assertAllClose(expected_output, actual_output)

    def test_output_dtypes(self):
        inputs = np.array([[[1], [2]], [[3], [4]]], dtype="float64")
        layer = RandomFlip()
        self.assertAllEqual(layer(inputs).dtype, "float32")
        layer = RandomFlip(dtype="uint8")
        self.assertAllEqual(layer(inputs).dtype, "uint8")

    def test_augment_bbox_batched_input(self):
        image = tf.zeros([20, 20, 3])
        bboxes = tf.convert_to_tensor(
            [
                [[0, 0, 10, 10], [4, 4, 12, 12]],
                [[0, 0, 10, 10], [4, 4, 12, 12]],
            ],
            dtype="float32",
        )
        input = {"images": [image, image], "bounding_boxes": bboxes}
        mock_random = [0.6, 0.6, 0.6, 0.6]
        layer = RandomFlip(bounding_box_format="xyxy")
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            side_effect=mock_random,
        ):
            output = layer(input, training=True)
        expected_output = [
            [[10, 10, 20, 20], [8, 8, 16, 16]],
            [[10, 10, 20, 20], [8, 8, 16, 16]],
        ]
        self.assertAllClose(expected_output, output["bounding_boxes"])
