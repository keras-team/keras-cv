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

from keras_cv.layers.preprocessing.equalization import Equalization


class EqualizationTest(tf.test.TestCase, parameterized.TestCase):
    def test_return_shapes(self):
        xs = 255 * tf.ones((2, 512, 512, 3), dtype=tf.int32)
        layer = Equalization(value_range=(0, 255))
        xs = layer(xs)

        self.assertEqual(xs.shape, [2, 512, 512, 3])
        self.assertAllEqual(xs, 255 * tf.ones((2, 512, 512, 3)))

    def test_return_shapes_inside_model(self):
        layer = Equalization(value_range=(0, 255))
        inp = tf.keras.layers.Input(shape=[512, 512, 5])
        out = layer(inp)
        model = tf.keras.models.Model(inp, out)

        self.assertEqual(model.layers[-1].output_shape, (None, 512, 512, 5))

    def test_equalizes_to_all_bins(self):
        xs = tf.random.uniform((2, 512, 512, 3), 0, 255, dtype=tf.float32)
        layer = Equalization(value_range=(0, 255))
        xs = layer(xs)

        for i in range(0, 256):
            self.assertTrue(tf.math.reduce_any(xs == i))

    @parameterized.named_parameters(
        ("float32", tf.float32), ("int32", tf.int32), ("int64", tf.int64)
    )
    def test_input_dtypes(self, dtype):
        xs = tf.random.uniform((2, 512, 512, 3), 0, 255, dtype=dtype)
        layer = Equalization(value_range=(0, 255))
        xs = layer(xs)

        for i in range(0, 256):
            self.assertTrue(tf.math.reduce_any(xs == i))
        self.assertAllInRange(xs, 0, 255)

    @parameterized.named_parameters(("0_255", 0, 255), ("0_1", 0, 1))
    def test_output_range(self, lower, upper):
        xs = tf.random.uniform((2, 512, 512, 3), lower, upper, dtype=tf.float32)
        layer = Equalization(value_range=(lower, upper))
        xs = layer(xs)
        self.assertAllInRange(xs, lower, upper)
