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

from keras_cv.layers import preprocessing


class RandomContrastTest(tf.test.TestCase):
    def test_preserves_output_shape(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = preprocessing.RandomContrast(factor=(0.3, 0.8))
        output = layer(image)

        self.assertEqual(image.shape, output.shape)
        self.assertNotAllClose(image, output)

    def test_no_adjustment_for_factor_zero(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = preprocessing.RandomContrast(factor=0)
        output = layer(image)

        self.assertAllClose(image, output, atol=1e-5, rtol=1e-5)

    def test_with_unit8(self):
        image_shape = (4, 8, 8, 3)
        image = tf.cast(tf.random.uniform(shape=image_shape) * 255.0, dtype=tf.uint8)

        layer = preprocessing.RandomContrast(factor=0)
        output = layer(image)
        self.assertAllClose(image, output, atol=1e-5, rtol=1e-5)

        layer = preprocessing.RandomContrast(factor=(0.3, 0.8))
        output = layer(image)
        self.assertNotAllClose(image, output)

    def test_config(self):
        layer = preprocessing.RandomContrast(factor=(0.3, 0.8))
        config = layer.get_config()
        self.assertEqual(config["factor"], (0.3, 0.8))
