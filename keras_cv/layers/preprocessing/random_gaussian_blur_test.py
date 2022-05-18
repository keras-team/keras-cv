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


class RandomGaussianBlurTest(tf.test.TestCase):
    def test_return_shapes(self):
        layer = preprocessing.RandomGaussianBlur(kernel_size=(3, 7), factor=(0, 2))

        # RGB
        xs = tf.ones((2, 512, 512, 3))
        xs = layer(xs)
        self.assertEqual(xs.shape, [2, 512, 512, 3])

        # greyscale
        xs = tf.ones((2, 512, 512, 1))
        xs = layer(xs)
        self.assertEqual(xs.shape, [2, 512, 512, 1])

    def test_in_single_image(self):
        layer = preprocessing.RandomGaussianBlur(kernel_size=(3, 7), factor=(0, 2))

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
        layer = preprocessing.RandomGaussianBlur(kernel_size=(3, 7), factor=(0, 2))

        # RGB
        xs = tf.ones((2, 256, 512, 3))
        xs = layer(xs)
        self.assertEqual(xs.shape, [2, 256, 512, 3])

        # greyscale
        xs = tf.ones((2, 256, 512, 1))
        xs = layer(xs)
        self.assertEqual(xs.shape, [2, 256, 512, 1])

    def test_single_input_args(self):
        layer = preprocessing.RandomGaussianBlur(kernel_size=7, factor=2)

        # RGB
        xs = tf.ones((2, 512, 512, 3))
        xs = layer(xs)
        self.assertEqual(xs.shape, [2, 512, 512, 3])

        # greyscale
        xs = tf.ones((2, 512, 512, 1))
        xs = layer(xs)
        self.assertEqual(xs.shape, [2, 512, 512, 1])

    def test_numerical(self):
        layer = preprocessing.RandomGaussianBlur(kernel_size=3, factor=(1.0, 1.0))

        xs = tf.expand_dims(
            tf.constant([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            axis=-1,
        )

        xs = tf.expand_dims(xs, axis=0)

        # Result expected to be identical to gaussian blur kernel of
        # size 3x3 and factor=1.0
        result = tf.expand_dims(
            tf.constant(
                [
                    [0.07511361, 0.1238414, 0.07511361],
                    [0.1238414, 0.20417996, 0.1238414],
                    [0.07511361, 0.1238414, 0.07511361],
                ]
            ),
            axis=-1,
        )
        result = tf.expand_dims(result, axis=0)
        xs = layer(xs)

        self.assertAllClose(xs, result)
