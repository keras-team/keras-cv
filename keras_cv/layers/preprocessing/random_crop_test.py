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

import unittest

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from keras_cv import layers as cv_layers
from keras_cv.layers.preprocessing.random_crop import RandomCrop
from keras_cv.tests.test_case import TestCase


class RandomCropTest(TestCase):
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
        resizing_layer = cv_layers.Resizing(height, width)
        expected_output = resizing_layer(inp)
        self.assertAllEqual(expected_output, actual_output)

    def test_training_with_mock(self):
        np.random.seed(1337)
        batch_size = 12
        height, width = 3, 4
        height_offset = np.random.randint(low=0, high=3)
        width_offset = np.random.randint(low=0, high=5)
        # manually compute transformations which shift height_offset and
        # width_offset respectively
        tops = np.ones((batch_size, 1)) * (height_offset / (5 - height))
        lefts = np.ones((batch_size, 1)) * (width_offset / (8 - width))
        transformations = {"tops": tops, "lefts": lefts}
        layer = RandomCrop(height, width)
        with unittest.mock.patch.object(
            layer,
            "get_random_transformation_batch",
            return_value=transformations,
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

    def test_unbatched_image(self):
        np.random.seed(1337)
        inp = np.random.random((16, 16, 3))
        # manually compute transformations which shift 2 pixels
        mock_offset = np.ones(shape=(1, 1), dtype="float32") * 0.25
        layer = RandomCrop(8, 8)
        with unittest.mock.patch.object(
            layer._random_generator,
            "uniform",
            return_value=mock_offset,
        ):
            actual_output = layer(inp, training=True)
            self.assertAllClose(inp[2:10, 2:10, :], actual_output)

    def test_batched_input(self):
        np.random.seed(1337)
        inp = np.random.random((20, 16, 16, 3))
        # manually compute transformations which shift 2 pixels
        mock_offset = np.ones(shape=(20, 1), dtype="float32") * 2 / (16 - 8)
        layer = RandomCrop(8, 8)
        with unittest.mock.patch.object(
            layer._random_generator,
            "uniform",
            return_value=mock_offset,
        ):
            actual_output = layer(inp, training=True)
            self.assertAllClose(inp[:, 2:10, 2:10, :], actual_output)

    def test_compute_ragged_output_signature(self):
        inputs = tf.ragged.stack(
            [
                np.random.random(size=(8, 8, 3)).astype("float32"),
                np.random.random(size=(16, 8, 3)).astype("float32"),
            ]
        )
        layer = RandomCrop(2, 2)
        output = layer(inputs)
        output_signature = layer.compute_ragged_image_signature(inputs).shape
        self.assertAllEqual(output.shape[1:], output_signature)

    def test_augment_bounding_boxes_crop(self):
        orig_height, orig_width = 512, 512
        height, width = 100, 200
        input_image = np.random.random((orig_height, orig_width, 3)).astype(
            np.float32
        )
        bboxes = {
            "boxes": np.array([[200, 200, 400, 400]]),
            "classes": np.array([1]),
        }
        input = {"images": input_image, "bounding_boxes": bboxes}
        # for top = 300 and left = 305
        height_offset = 300
        width_offset = 305
        tops = np.ones((1, 1)) * (height_offset / (orig_height - height))
        lefts = np.ones((1, 1)) * (width_offset / (orig_width - width))
        transformations = {"tops": tops, "lefts": lefts}
        layer = RandomCrop(
            height=height, width=width, bounding_box_format="xyxy"
        )
        with unittest.mock.patch.object(
            layer,
            "get_random_transformation_batch",
            return_value=transformations,
        ):
            output = layer(input)
            expected_output = np.asarray(
                [[0.0, 0.0, 95.0, 100.0]],
            )
        self.assertAllClose(expected_output, output["bounding_boxes"]["boxes"])

    def test_augment_bounding_boxes_resize(self):
        input_image = np.random.random((256, 256, 3)).astype(np.float32)
        bboxes = {
            "boxes": np.array([[100, 100, 200, 200]]),
            "classes": np.array([1]),
        }
        input = {"images": input_image, "bounding_boxes": bboxes}
        layer = RandomCrop(height=512, width=512, bounding_box_format="xyxy")
        output = layer(input)
        expected_output = np.asarray(
            [[200.0, 200.0, 400.0, 400.0]],
        )
        self.assertAllClose(expected_output, output["bounding_boxes"]["boxes"])

    def test_in_tf_function(self):
        np.random.seed(1337)
        inp = np.random.random((20, 16, 16, 3))
        mock_offset = np.ones(shape=(20, 1), dtype="float32") * 2 / (16 - 8)
        layer = RandomCrop(8, 8)

        @tf.function
        def augment(x):
            return layer(x, training=True)

        with unittest.mock.patch.object(
            layer._random_generator,
            "uniform",
            return_value=mock_offset,
        ):
            actual_output = augment(inp)
            self.assertAllClose(inp[:, 2:10, 2:10, :], actual_output)

    def test_random_crop_on_batched_images_independently(self):
        image = tf.random.uniform((100, 100, 3))
        batched_images = tf.stack((image, image), axis=0)
        layer = RandomCrop(height=25, width=25)

        results = layer(batched_images)

        self.assertNotAllClose(results[0], results[1])

    def test_random_crop_on_batched_ragged_images_and_bounding_boxes(self):
        images = tf.ragged.constant(
            [np.ones((8, 8, 3)), np.ones((4, 8, 3))], dtype="float32"
        )
        boxes = {
            "boxes": tf.ragged.stack(
                [
                    np.ones((3, 4), dtype="float32"),
                    np.ones((3, 4), dtype="float32"),
                ],
            ),
            "classes": tf.ragged.stack(
                [
                    np.ones((3,), dtype="float32"),
                    np.ones((3,), dtype="float32"),
                ],
            ),
        }
        inputs = {"images": images, "bounding_boxes": boxes}
        layer = RandomCrop(height=2, width=2, bounding_box_format="xyxy")

        results = layer(inputs)

        self.assertTrue(isinstance(results["images"], tf.Tensor))
        self.assertTrue(
            isinstance(results["bounding_boxes"]["boxes"], tf.RaggedTensor)
        )
        self.assertTrue(
            isinstance(results["bounding_boxes"]["classes"], tf.RaggedTensor)
        )

    def test_config_with_custom_name(self):
        layer = RandomCrop(5, 5, name="image_preproc")
        config = layer.get_config()
        layer_1 = RandomCrop.from_config(config)
        self.assertEqual(layer_1.name, layer.name)

    def test_output_dtypes(self):
        inputs = np.array([[[1], [2]], [[3], [4]]], dtype="float64")
        layer = RandomCrop(2, 2)
        self.assertAllEqual(layer(inputs).dtype, "float32")
        layer = RandomCrop(2, 2, dtype="uint8")
        self.assertAllEqual(layer(inputs).dtype, "uint8")

    def test_config(self):
        layer = RandomCrop(height=2, width=3, bounding_box_format="xyxy")
        config = layer.get_config()
        self.assertEqual(config["height"], 2)
        self.assertEqual(config["width"], 3)
        self.assertEqual(config["bounding_box_format"], "xyxy")
