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

import keras_cv
from keras_cv.layers.preprocessing.grid_mask import GridMask


class GridMaskTest(tf.test.TestCase):
    def test_return_shapes(self):
        xs = tf.ones((2, 512, 512, 3))

        layer = GridMask(ratio_factor=0.1, rotation_factor=(-0.2, 0.3))
        xs = layer(xs, training=True)

        self.assertEqual(xs.shape, [2, 512, 512, 3])

    def test_gridmask_call_results_one_channel(self):
        xs = tf.cast(
            tf.stack(
                [3 * tf.ones((40, 40, 1)), 2 * tf.ones((40, 40, 1))],
                axis=0,
            ),
            dtype=tf.float32,
        )

        fill_value = 0.0
        layer = GridMask(
            ratio_factor=0.3,
            rotation_factor=(0.2, 0.3),
            fill_mode="constant",
            fill_value=fill_value,
        )
        xs = layer(xs, training=True)

        # Some pixels should be replaced with fill_value
        self.assertTrue(tf.math.reduce_any(xs[0] == float(fill_value)))
        self.assertTrue(tf.math.reduce_any(xs[0] == 3.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == float(fill_value)))
        self.assertTrue(tf.math.reduce_any(xs[1] == 2.0))

    def test_non_square_image(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((1024, 512, 1)), tf.ones((1024, 512, 1))],
                axis=0,
            ),
            dtype=tf.float32,
        )

        fill_value = 100.0
        layer = GridMask(
            ratio_factor=0.6,
            rotation_factor=0.3,
            fill_mode="constant",
            fill_value=fill_value,
        )
        xs = layer(xs, training=True)

        # Some pixels should be replaced with fill_value
        self.assertTrue(tf.math.reduce_any(xs[0] == float(fill_value)))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == float(fill_value)))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_in_tf_function(self):
        xs = tf.cast(
            tf.stack([2 * tf.ones((100, 100, 1)), tf.ones((100, 100, 1))], axis=0),
            dtype=tf.float32,
        )

        fill_value = 255.0
        layer = GridMask(
            ratio_factor=keras_cv.ConstantFactorSampler(0.5),
            rotation_factor=0.5,
            fill_mode="constant",
            fill_value=fill_value,
        )

        @tf.function
        def augment(x):
            return layer(x, training=True)

        xs = augment(xs)

        # Some pixels should be replaced with fill_value
        self.assertTrue(tf.math.reduce_any(xs[0] == float(fill_value)))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == float(fill_value)))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_in_single_image(self):
        xs = tf.cast(
            tf.ones((512, 512, 1)),
            dtype=tf.float32,
        )

        layer = GridMask(ratio_factor=(0.5, 0.5), fill_mode="constant", fill_value=0.0)
        xs = layer(xs, training=True)
        self.assertTrue(tf.math.reduce_any(xs == 0.0))
        self.assertTrue(tf.math.reduce_any(xs == 1.0))
