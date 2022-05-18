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


class RandomShearTest(tf.test.TestCase):
    def test_aggressive_shear_fills_at_least_some_pixels(self):
        img_shape = (50, 50, 3)
        xs = tf.stack(
            [2 * tf.ones(img_shape), tf.ones(img_shape)],
            axis=0,
        )
        xs = tf.cast(xs, tf.float32)

        fill_value = 0.0
        layer = preprocessing.RandomShear(
            x_factor=(3, 3), seed=0, fill_mode="constant", fill_value=fill_value
        )
        xs = layer(xs)

        # Some pixels should be replaced with fill value
        self.assertTrue(tf.math.reduce_any(xs[0] == fill_value))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == fill_value))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))
