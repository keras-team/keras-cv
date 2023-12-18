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

from keras_cv import core
from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.layers import preprocessing
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.tests.test_case import TestCase
from keras_cv.utils import preprocessing as preprocessing_utils


class OldRandomSaturation(BaseImageAugmentationLayer):
    """Randomly adjusts the saturation on given images.

    This layer will randomly increase/reduce the saturation for the input RGB
    images. At inference time, the output will be identical to the input.
    Call the layer with `training=True` to adjust the saturation of the input.

    Args:
        factor: A tuple of two floats, a single float or
            `keras_cv.FactorSampler`. `factor` controls the extent to which the
            image saturation is impacted. `factor=0.5` makes this layer perform
            a no-op operation. `factor=0.0` makes the image to be fully
            grayscale. `factor=1.0` makes the image to be fully saturated.
            Values should be between `0.0` and `1.0`. If a tuple is used, a
            `factor` is sampled between the two values for every image
            augmented. If a single float is used, a value between `0.0` and the
            passed float is sampled. In order to ensure the value is always the
            same, please pass a tuple with two identical floats: `(0.5, 0.5)`.
        seed: Integer. Used to create a random seed.
    """

    def __init__(self, factor, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.factor = preprocessing_utils.parse_factor(
            factor,
            min_value=0.0,
            max_value=1.0,
        )
        self.seed = seed

    def get_random_transformation(self, **kwargs):
        return self.factor()

    def augment_image(self, image, transformation=None, **kwargs):
        # Convert the factor range from [0, 1] to [0, +inf]. Note that the
        # tf.image.adjust_saturation is trying to apply the following math
        # formula `output_saturation = input_saturation * factor`. We use the
        # following method to the do the mapping.
        # `y = x / (1 - x)`.
        # This will ensure:
        #   y = +inf when x = 1 (full saturation)
        #   y = 1 when x = 0.5 (no augmentation)
        #   y = 0 when x = 0 (full gray scale)

        # Convert the transformation to tensor in case it is a float. When
        # transformation is 1.0, then it will result in to divide by zero error,
        # but it will be handled correctly when it is a one tensor.
        transformation = tf.convert_to_tensor(transformation)
        adjust_factor = transformation / (1 - transformation)
        return tf.image.adjust_saturation(
            image, saturation_factor=adjust_factor
        )

    def augment_bounding_boxes(
        self, bounding_boxes, transformation=None, **kwargs
    ):
        return bounding_boxes

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def augment_segmentation_mask(
        self, segmentation_mask, transformation, **kwargs
    ):
        return segmentation_mask

    def get_config(self):
        config = {
            "factor": self.factor,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if isinstance(config["factor"], dict):
            config["factor"] = keras.utils.deserialize_keras_object(
                config["factor"]
            )
        return cls(**config)


class RandomSaturationTest(TestCase):
    def test_preserves_output_shape(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = preprocessing.RandomSaturation(factor=(0.3, 0.8))
        output = layer(image)

        self.assertEqual(image.shape, output.shape)
        self.assertNotAllClose(image, output)

    def test_no_adjustment_for_factor_point_five(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = preprocessing.RandomSaturation(factor=(0.5, 0.5))
        output = layer(image)

        self.assertAllClose(image, output, atol=1e-5, rtol=1e-5)

    def test_adjust_to_grayscale(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = preprocessing.RandomSaturation(factor=(0.0, 0.0))
        output = ops.convert_to_numpy(layer(image))

        channel_mean = np.mean(output, axis=-1)
        channel_values = tf.unstack(output, axis=-1)
        # Make sure all the pixel has the same value among the channel dim,
        # which is a fully gray RGB.
        for channel_value in channel_values:
            self.assertAllClose(
                channel_mean, channel_value, atol=1e-5, rtol=1e-5
            )

    def test_adjust_to_full_saturation(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = preprocessing.RandomSaturation(factor=(1.0, 1.0))
        output = ops.convert_to_numpy(layer(image))

        channel_mean = np.min(output, axis=-1)
        # Make sure at least one of the channel is 0.0 (fully saturated image)
        self.assertAllClose(channel_mean, np.zeros((4, 8, 8)))

    def test_adjustment_for_non_rgb_value_range(self):
        image_shape = (4, 8, 8, 3)
        # Value range (0, 100)
        image = tf.random.uniform(shape=image_shape) * 100.0

        layer = preprocessing.RandomSaturation(factor=(0.5, 0.5))
        output = layer(image)
        self.assertAllClose(image, output, atol=1e-5, rtol=1e-5)

        layer = preprocessing.RandomSaturation(factor=(0.3, 0.8))
        output = layer(image)
        self.assertNotAllClose(image, output)

    def test_with_unit8(self):
        image_shape = (4, 8, 8, 3)
        image = tf.cast(
            tf.random.uniform(shape=image_shape) * 255.0, dtype=tf.uint8
        )

        layer = preprocessing.RandomSaturation(factor=(0.5, 0.5))
        output = layer(image)
        self.assertAllClose(image, output, atol=1e-5, rtol=1e-5)

        layer = preprocessing.RandomSaturation(factor=(0.3, 0.8))
        output = layer(image)
        self.assertNotAllClose(image, output)

    def test_config(self):
        layer = preprocessing.RandomSaturation(factor=(0.3, 0.8))
        config = layer.get_config()
        self.assertTrue(isinstance(config["factor"], core.UniformFactorSampler))
        self.assertEqual(config["factor"].get_config()["lower"], 0.3)
        self.assertEqual(config["factor"].get_config()["upper"], 0.8)

    def test_correctness_with_tf_adjust_saturation_normalized_range(self):
        image_shape = (16, 32, 32, 3)
        fixed_factor = (0.8, 0.8)
        image = tf.random.uniform(shape=image_shape)

        layer = preprocessing.RandomSaturation(factor=fixed_factor)
        old_layer = OldRandomSaturation(factor=fixed_factor)

        output = layer(image)
        old_output = old_layer(image)

        self.assertAllClose(old_output, output, atol=1e-5, rtol=1e-5)

    def test_correctness_with_tf_adjust_saturation_rgb_range(self):
        image_shape = (16, 32, 32, 3)
        fixed_factor = (0.8, 0.8)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = preprocessing.RandomSaturation(factor=fixed_factor)
        old_layer = OldRandomSaturation(factor=fixed_factor)

        output = layer(image)
        old_output = old_layer(image)

        self.assertAllClose(old_output, output, atol=1e-3, rtol=1e-5)
