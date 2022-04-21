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

from keras_cv.layers.preprocessing.gaussian_blur import GaussianBlur


class GaussianBlurTest(tf.test.TestCase):
    def test_return_shapes(self):
        # RGB images
        xs_rgb = tf.ones((2, 512, 512, 3))
        # grayscale images
        xs_g = tf.ones((2, 512, 512, 1))

        layer = GaussianBlur(kernel_size=(7, 3), sigma=1)
        xs1 = layer(xs_rgb, training=True)
        xs2 = layer(xs_g, training=True)

        self.assertEqual(xs1.shape, [2, 512, 512, 3])
        self.assertEqual(xs2.shape, [2, 512, 512, 1])

    def test_in_tf_function(self):
        xs = tf.cast(
            tf.stack([2 * tf.ones((100, 100, 3)), tf.ones((100, 100, 3))], axis=0),
            tf.float32,
        )

        # test 1
        layer = GaussianBlur(kernel_size=(7, 3), sigma=1)

        @tf.function
        def augment(x):
            return layer(x, training=True)

        xs1 = augment(xs)

        self.assertEqual(xs1.shape, [2, 100, 100, 3])

    def test_non_square_image(self):
        xs = tf.cast(
            tf.stack([2 * tf.ones((512, 1024, 3)), tf.ones((512, 1024, 3))], axis=0),
            tf.float32,
        )

        layer = GaussianBlur(kernel_size=(7, 3), sigma=1)
        xs1 = layer(xs, training=True)

        self.assertEqual(xs1.shape, [2, 512, 1024, 3])

    def test_in_single_image(self):
        xs = tf.cast(
            tf.ones((512, 512, 3)),
            dtype=tf.float32,
        )

        layer = GaussianBlur(kernel_size=(7, 3), sigma=1)
        xs1 = layer(xs, training=True)

        self.assertEqual(xs1.shape, [512, 512, 3])
