import unittest

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from keras_cv.layers.preprocessing.random_width import RandomWidth

class RandomWidthTest(tf.test.TestCase, parameterized.TestCase):
    def _run_test(self, factor):
        np.random.seed(1337)
        num_samples = 2
        orig_height = 5
        orig_width = 8
        channels = 3
        img = np.random.random(
            (num_samples, orig_height, orig_width, channels)
        )
        layer = RandomWidth(factor)
        img_out = layer(img, training=True)
        self.assertEqual(img_out.shape[0], 2)
        self.assertEqual(img_out.shape[1], 5)
        self.assertEqual(img_out.shape[3], 3)

    @parameterized.named_parameters(
        ("random_width_4_by_6", (0.4, 0.6)),
        ("random_width_3_by_2", (-0.3, 0.2)),
        ("random_width_3", 0.3),
    )
    def test_random_width_basic(self, factor):
        self._run_test(factor)

    def test_valid_random_width(self):
        # need (maxval - minval) * rnd + minval = 0.6
        mock_factor = 0.6
        img = np.random.random((12, 8, 5, 3))
        layer = RandomWidth(0.4)
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            return_value=mock_factor
        ):
            img_out = layer(img, training=True)
            self.assertEqual(img_out.shape[2], 3)

    def test_random_width_longer_numeric(self):
        for dtype in (np.int64, np.float32):
            input_image = np.reshape(np.arange(0, 6), (3, 2, 1)).astype(
                dtype
            )
            layer = RandomWidth(factor=(1.0, 1.0))
            # Return type of RandomWidth() is float32 if `interpolation` is
            # not set to `ResizeMethod.NEAREST_NEIGHBOR`; cast `layer` to
            # desired dtype.
            output_image = tf.cast(
                layer(np.expand_dims(input_image, axis=0)), dtype=dtype
            )
            # pyformat: disable
            expected_output = np.asarray(
                [[0, 0.25, 0.75, 1], [2, 2.25, 2.75, 3], [4, 4.25, 4.75, 5]]
            ).astype(dtype)
            # pyformat: enable
            expected_output = np.reshape(expected_output, (1, 3, 4, 1))
            self.assertAllEqual(expected_output, output_image)

    def test_random_width_shorter_numeric(self):
        for dtype in (np.int64, np.float32):
            input_image = np.reshape(np.arange(0, 8), (2, 4, 1)).astype(
                dtype
            )
            layer = RandomWidth(
                factor=(-0.5, -0.5), interpolation="nearest"
            )
            output_image = layer(np.expand_dims(input_image, axis=0))
            # pyformat: disable
            expected_output = np.asarray([[1, 3], [5, 7]]).astype(dtype)
            # pyformat: enable
            expected_output = np.reshape(expected_output, (1, 2, 2, 1))
            self.assertAllEqual(expected_output, output_image)

    def test_random_width_invalid_factor(self):
        with self.assertRaises(ValueError):
            RandomWidth((-1.5, 0.4))

    def test_random_width_inference(self):
        input_images = np.random.random((2, 5, 8, 3)).astype(np.float32)
        expected_output = input_images
        layer =RandomWidth(0.5)
        actual_output = layer(input_images, training=False)
        self.assertAllClose(expected_output, actual_output)

    def test_config_with_custom_name(self):
        layer = RandomWidth(0.5, name="image_preproc")
        config = layer.get_config()
        layer_1 = RandomWidth.from_config(config)
        print(type(layer))
        print(type(layer_1))
        self.assertEqual(layer_1.name, layer.name)

    def test_unbatched_image(self):
        # need (maxval - minval) * rnd + minval = 0.6
        mock_factor = 0.6
        img = np.random.random((8, 5, 3))
        layer = RandomWidth(0.4)
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            return_value=mock_factor,
        ):
            img_out = layer(img, training=True)
            self.assertEqual(img_out.shape[1], 3)

    def test_batched_input(self):
        # need (maxval - minval) * rnd + minval = 0.6
        mock_factor = 0.6
        img = np.random.random((12, 8, 5, 3))
        layer = RandomWidth(0.4)
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            return_value=mock_factor,
        ):
            img_out = layer(img, training=True)
            self.assertEqual(img_out.shape[2], 3)

    def test_augment_image(self):
        # need (maxval - minval) * rnd + minval = 0.6
        mock_factor = 0.6
        img = np.random.random((8, 5, 3))
        layer = RandomWidth(0.4)
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            return_value=mock_factor,
        ):
            img_out = layer.augment_image(
                img,
                transformation=layer.get_random_transformation(image=img),
            )
            self.assertEqual(img_out.shape[1], 3)

    def test_output_dtypes(self):
        inputs = np.array([[[1], [2]], [[3], [4]]], dtype="float64")
        layer = RandomWidth(0.2)
        self.assertAllEqual(layer(inputs).dtype, "float32")
        layer = RandomWidth(0.2, dtype="uint8")
        self.assertAllEqual(layer(inputs).dtype, "uint8")