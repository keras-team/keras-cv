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

from keras_cv.layers import preprocessing


class RandomBrightnessTest(tf.test.TestCase):

    def test_scale_input_validation(self):
      with self.assertRaisesRegexp(ValueError, 'ranged between [-1.0, 1.0]'):
        preprocessing.RandomBrightness(2.0)

      with self.assertRaisesRegexp(ValueError, 'list of two numbers'):
        preprocessing.RandomBrightness([1.0])

      with self.assertRaisesRegexp(ValueError, 'should be number'):
        preprocessing.RandomBrightness('one')

    def test_scale_normalize(self):
        layer = preprocessing.RandomBrightness(1.0)
        self.assertEqual(layer._scale, [-1.0, 1.0])

        layer = preprocessing.RandomBrightness((0.5, 0.3))
        self.assertEqual(layer._scale, [0.3, 0.5])

        layer = preprocessing.RandomBrightness(-0.2)
        self.assertEqual(layer._scale, [-0.2, 0.2])

    def test_output_value_range(self):
        # Always scale up to 255
        layer = preprocessing.RandomBrightness([1.0, 1.0])
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=True)
        output_min = tf.math.reduce_min(output)
        output_max = tf.math.reduce_max(output)
        self.assertEqual(output_min, 255)
        self.assertEqual(output_max, 255)

        # Always scale down to 0
        layer = preprocessing.RandomBrightness([-1.0, -1.0])
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=True)
        output_min = tf.math.reduce_min(output)
        output_max = tf.math.reduce_max(output)
        self.assertEqual(output_min, 0)
        self.assertEqual(output_max, 0)

    def test_output(self):
        # Always scale up, but randomly between 0 ~ 255
        layer = preprocessing.RandomBrightness([0, 1.0])
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=True)
        diff = output - inputs
        self.assertGreaterEqual(tf.math.reduce_min(diff), 0)
        self.assertGreater(tf.math.reduce_mean(diff), 0)

        # Always scale down, but randomly between 0 ~ 255
        layer = preprocessing.RandomBrightness([-1.0, 0.0])
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=True)
        diff = output - inputs
        self.assertLessEqual(tf.math.reduce_max(diff), 0)
        self.assertLess(tf.math.reduce_mean(diff), 0)

    def test_same_adjustment_within_batch(self):
        layer = preprocessing.RandomBrightness([0.2, 0.3])
        inputs = np.zeros(shape=(2, 224, 224, 3))   # 2 images with all zeros
        output = layer(inputs, training=True)
        diff = output - inputs
        # Make sure two images gets the same adjustment
        self.assertAllClose(diff[0], diff[1])

    def test_inference(self):
        layer = preprocessing.RandomBrightness([0, 1.0])
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs)
        self.assertAllClose(inputs, output)

        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output)

    def test_dtype(self):
        layer = preprocessing.RandomBrightness([0, 1.0])
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=True)
        self.assertEqual(output.dtype, tf.int64)

        inputs = tf.cast(inputs, tf.float32)
        output = layer(inputs, training=True)
        self.assertEqual(output.dtype, tf.float32)

    def test_seed(self):
        layer = preprocessing.RandomBrightness([0, 1.0], seed=1337)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output_1 = layer(inputs, training=True)
        output_2 = layer(inputs, training=True)

        self.assertAllClose(output_1, output_2)

    def test_config(self):
        layer = preprocessing.RandomBrightness([0, 1.0], seed=1337)
        config = layer.get_config()
        self.assertEqual(config['scale'], [0.0, 1.0])
        self.assertEqual(config['seed'], 1337)

        reconstructed_layer = preprocessing.RandomBrightness.from_config(config)
        self.assertEqual(reconstructed_layer._scale, layer._scale)
        self.assertEqual(reconstructed_layer._seed, layer._seed)