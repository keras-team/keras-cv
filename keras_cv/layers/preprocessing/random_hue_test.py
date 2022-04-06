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
import tensorflow as tf
from absl.testing import parameterized

from keras_cv import core
from keras_cv.layers import preprocessing


class RandomHueTest(tf.test.TestCase, parameterized.TestCase):
    def test_preserves_output_shape(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = preprocessing.RandomHue(factor=(0.3, 0.8), value_range=(0, 255))
        output = layer(image)

        self.assertEqual(image.shape, output.shape)
        self.assertNotAllClose(image, output)

    def test_adjust_no_op(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = preprocessing.RandomHue(factor=(0.0, 0.0), value_range=(0, 255))
        output = layer(image)
        self.assertAllClose(image, output, atol=1e-5, rtol=1e-5)

    def test_adjust_full_opposite_hue(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = preprocessing.RandomHue(factor=(1.0, 1.0), value_range=(0, 255))
        output = layer(image)

        channel_max = tf.math.reduce_max(output, axis=-1)
        channel_min = tf.math.reduce_min(output, axis=-1)
        # Make sure the max and min channel are the same between input and output
        # In the meantime, and channel will swap between each other.
        self.assertAllClose(channel_max, tf.math.reduce_max(image, axis=-1))
        self.assertAllClose(channel_min, tf.math.reduce_min(image, axis=-1))

    @parameterized.named_parameters(
        ("025", 0.25), ("05", 0.5), ("075", 0.75), ("100", 1.0)
    )
    def test_adjusts_all_values_for_factor(self, factor):
        image_shape = (4, 8, 8, 3)
        # Value range (0, 100)
        image = tf.random.uniform(shape=image_shape) * 100.0

        layer = preprocessing.RandomHue(factor=(factor, factor), value_range=(0, 255))
        output = layer(image)
        self.assertNotAllClose(image, output, atol=1e-5, rtol=1e-5)

    def test_adjustment_for_non_rgb_value_range(self):
        image_shape = (4, 8, 8, 3)
        # Value range (0, 100)
        image = tf.random.uniform(shape=image_shape) * 100.0

        layer = preprocessing.RandomHue(factor=(0.0, 0.0), value_range=(0, 255))
        output = layer(image)
        self.assertAllClose(image, output, atol=1e-5, rtol=1e-5)

        layer = preprocessing.RandomHue(factor=(0.3, 0.8), value_range=(0, 255))
        output = layer(image)
        self.assertNotAllClose(image, output)

    def test_with_uint8(self):
        image_shape = (4, 8, 8, 3)
        image = tf.cast(tf.random.uniform(shape=image_shape) * 255.0, dtype=tf.uint8)

        layer = preprocessing.RandomHue(factor=(0.0, 0.0), value_range=(0, 255))
        output = layer(image)
        self.assertAllClose(image, output, atol=1e-5, rtol=1e-5)

        layer = preprocessing.RandomHue(factor=(0.3, 0.8), value_range=(0, 255))
        output = layer(image)
        self.assertNotAllClose(image, output)

    def test_config(self):
        layer = preprocessing.RandomHue(factor=(0.3, 0.8), value_range=(0, 255))
        config = layer.get_config()
        self.assertTrue(isinstance(config["factor"], core.UniformFactorSampler))
        self.assertEqual(config["factor"].get_config()["lower"], 0.3)
        self.assertEqual(config["factor"].get_config()["upper"], 0.8)
        self.assertEqual(config["value_range"], (0, 255))
