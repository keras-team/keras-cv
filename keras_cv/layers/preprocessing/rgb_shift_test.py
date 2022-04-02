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

from keras_cv.layers.preprocessing.rgb_shift import RGBShift


class RGBShiftTest(tf.test.TestCase):
    def test_return_transformation(self):
        factor_values = (0.2, 0.2)
        layer = RGBShift(factor=factor_values, value_range=(0, 255))
        xs = layer.get_random_transformation()
        xs = tf.concat(xs, axis=0)
        self.assertTrue(tf.math.reduce_any(xs == factor_values[0]))
        self.assertTrue(tf.math.reduce_any(xs == factor_values[1]))

        factor_values = 0.2
        layer = RGBShift(factor=factor_values, value_range=(0, 255))
        xs = layer.get_random_transformation()
        xs = tf.concat(xs, axis=0)
        self.assertFalse(tf.math.reduce_any(xs == factor_values))
        self.assertFalse(tf.math.reduce_any(xs == factor_values))

    def test_return_shapes(self):
        xs = tf.ones((2, 512, 512, 3))
        layer = RGBShift(factor=1.0, value_range=(0, 255))

        xs = layer(xs, training=True)
        self.assertEqual(xs.shape, [2, 512, 512, 3])

    def test_non_square_image(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((1024, 512, 3)), tf.ones((1024, 512, 3))],
                axis=0,
            ),
            dtype=tf.float32,
        )
        layer = RGBShift(factor=[0.1, 0.3], value_range=(0, 255))

        xs = layer(xs, training=True)
        self.assertFalse(tf.math.reduce_any(xs[0] == 2.0))
        self.assertFalse(tf.math.reduce_any(xs[1] == 1.0))

    def test_in_tf_function(self):
        xs = tf.cast(
            tf.stack([2 * tf.ones((100, 100, 3)), tf.ones((100, 100, 3))], axis=0),
            dtype=tf.float32,
        )
        layer = RGBShift(factor=0.3, value_range=(0, 255))

        @tf.function
        def augment(x):
            return layer(x, training=True)

        xs = augment(xs)
        self.assertFalse(tf.math.reduce_any(xs[0] == 2.0))
        self.assertFalse(tf.math.reduce_any(xs[1] == 1.0))

    def test_in_single_image(self):
        xs = tf.cast(
            tf.ones((512, 512, 3)),
            dtype=tf.float32,
        )
        layer = RGBShift(factor=0.4, value_range=(0, 255))
        xs = layer(xs, training=True)
        self.assertFalse(tf.math.reduce_any(xs == 1.0))

    def test_config(self):
        layer = RGBShift(factor=[0.1, 0.5], value_range=(0, 255))
        config = layer.get_config()
        self.assertEqual(config["factor"], [0.1, 0.5])

        reconstructed_layer = RGBShift.from_config(config)
        self.assertEqual(reconstructed_layer.factor, layer.factor)

    def test_inference(self):
        layer = RGBShift(factor=0.8, value_range=(0, 255))
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output)
