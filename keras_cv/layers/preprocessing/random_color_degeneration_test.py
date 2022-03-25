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


class RandomColorDegenerationTest(tf.test.TestCase):
    def test_random_color_degeneration_base_case(self):
        img_shape = (50, 50, 3)
        xs = tf.stack(
            [2 * tf.ones(img_shape), tf.ones(img_shape)],
            axis=0,
        )

        layer = preprocessing.RandomColorDegeneration(0.0)
        ys = layer(xs)

        self.assertEqual(xs.shape, ys.shape)

    def test_color_degeneration_full_factor(self):
        img_shape = (50, 50, 1)
        r = tf.ones(img_shape)
        g = 2 * tf.ones(img_shape)
        b = 3 * tf.ones(img_shape)
        xs = tf.concat([r, g, b], axis=-1)

        layer = preprocessing.RandomColorDegeneration(factor=(1, 1))
        ys = layer(xs)

        # Color degeneration uses standard luma conversion for RGB->Grayscale.
        # The formula for luma is result= 0.2989*r + 0.5870*g + 0.1140*b
        luma_result = 0.2989 + 2 * 0.5870 + 3 * 0.1140
        self.assertAllClose(ys, tf.ones_like(ys) * luma_result)

    def test_color_degeneration_70p_factor(self):
        img_shape = (50, 50, 1)
        r = tf.ones(img_shape)
        g = 2 * tf.ones(img_shape)
        b = 3 * tf.ones(img_shape)
        xs = tf.concat([r, g, b], axis=-1)

        layer = preprocessing.RandomColorDegeneration(factor=(0.7, 0.7))
        ys = layer(xs)

        # Color degeneration uses standard luma conversion for RGB->Grayscale.
        # The formula for luma is result= 0.2989*r + 0.5870*g + 0.1140*b
        luma_result = 0.2989 + 2 * 0.5870 + 3 * 0.1140

        # with factor=0.7, luma_result should be blended at a 70% rate with the original
        r_result = luma_result * 0.7 + 1 * 0.3
        g_result = luma_result * 0.7 + 2 * 0.3
        b_result = luma_result * 0.7 + 3 * 0.3

        r = ys[..., 0]
        g = ys[..., 1]
        b = ys[..., 2]

        self.assertAllClose(r, tf.ones_like(r) * r_result)
        self.assertAllClose(g, tf.ones_like(g) * g_result)
        self.assertAllClose(b, tf.ones_like(b) * b_result)
