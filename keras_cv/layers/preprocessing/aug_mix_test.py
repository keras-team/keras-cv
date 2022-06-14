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


class AugMixTest(tf.test.TestCase):
    def test_return_shapes(self):
        layer = preprocessing.AugMix([0, 255])

        # RGB
        xs = tf.ones((2, 512, 512, 3))
        xs = layer(xs)
        self.assertEqual(xs.shape, [2, 512, 512, 3])

        # greyscale
        xs = tf.ones((2, 512, 512, 1))
        xs = layer(xs)
        self.assertEqual(xs.shape, [2, 512, 512, 1])

    def test_in_single_image(self):
        layer = preprocessing.AugMix([0, 255])

        # RGB
        xs = tf.cast(
            tf.ones((512, 512, 3)),
            dtype=tf.float32,
        )

        xs = layer(xs)
        self.assertEqual(xs.shape, [512, 512, 3])

        # greyscale
        xs = tf.cast(
            tf.ones((512, 512, 1)),
            dtype=tf.float32,
        )

        xs = layer(xs)
        self.assertEqual(xs.shape, [512, 512, 1])

    def test_non_square_images(self):
        layer = preprocessing.AugMix([0, 255])

        # RGB
        xs = tf.ones((2, 256, 512, 3))
        xs = layer(xs)
        self.assertEqual(xs.shape, [2, 256, 512, 3])

        # greyscale
        xs = tf.ones((2, 256, 512, 1))
        xs = layer(xs)
        self.assertEqual(xs.shape, [2, 256, 512, 1])

    def test_single_input_args(self):
        layer = preprocessing.AugMix([0, 255])

        # RGB
        xs = tf.ones((2, 512, 512, 3))
        xs = layer(xs)
        self.assertEqual(xs.shape, [2, 512, 512, 3])

        # greyscale
        xs = tf.ones((2, 512, 512, 1))
        xs = layer(xs)
        self.assertEqual(xs.shape, [2, 512, 512, 1])
