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


class RandomCutoutTest(tf.test.TestCase):
    def _run_test(self, height_factor, width_factor):
        img_shape = (40, 40, 3)
        xs = tf.stack(
            [2 * tf.ones(img_shape), tf.ones(img_shape)],
            axis=0,
        )
        xs = tf.cast(xs, tf.float32)

        fill_value = 0.0
        layer = preprocessing.RandomCutout(
            height_factor=height_factor,
            width_factor=width_factor,
            fill_mode="constant",
            fill_value=fill_value,
            seed=1,
        )
        xs = layer(xs)

        # Some pixels should be replaced with fill value
        self.assertTrue(tf.math.reduce_any(xs[0] == fill_value))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == fill_value))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_return_shapes(self):
        xs = tf.ones((2, 512, 512, 3))

        layer = preprocessing.RandomCutout(height_factor=0.5, width_factor=0.5, seed=1)
        xs = layer(xs)

        self.assertEqual(xs.shape, [2, 512, 512, 3])

    def test_return_shapes_single_element(self):
        xs = tf.ones((512, 512, 3))

        layer = preprocessing.RandomCutout(height_factor=0.5, width_factor=0.5, seed=1)
        xs = layer(xs)

        self.assertEqual(xs.shape, [512, 512, 3])

    def test_random_cutout_single_float(self):
        self._run_test(0.5, 0.5)

    def test_random_cutout_tuple_float(self):
        self._run_test((0.4, 0.9), (0.1, 0.3))

    def test_random_cutout_fail_mix_bad_param_values(self):
        fn = lambda: self._run_test(0.5, (15.0, 30))
        self.assertRaises(ValueError, fn)

    def test_random_cutout_fail_reverse_lower_upper_float(self):
        fn = lambda: self._run_test(0.5, (0.9, 0.4))
        self.assertRaises(ValueError, fn)

    def test_random_cutout_call_results_one_channel(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((40, 40, 1)), tf.ones((40, 40, 1))],
                axis=0,
            ),
            tf.float32,
        )

        patch_value = 0.0
        layer = preprocessing.RandomCutout(
            height_factor=0.5,
            width_factor=0.5,
            fill_mode="constant",
            fill_value=patch_value,
            seed=1,
        )
        xs = layer(xs)

        # Some pixels should be replaced with fill value
        self.assertTrue(tf.math.reduce_any(xs[0] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_random_cutout_call_tiny_image(self):
        img_shape = (4, 4, 3)
        xs = tf.stack(
            [2 * tf.ones(img_shape), tf.ones(img_shape)],
            axis=0,
        )
        xs = tf.cast(xs, tf.float32)

        fill_value = 0.0
        layer = preprocessing.RandomCutout(
            height_factor=(0.4, 0.9),
            width_factor=(0.1, 0.3),
            fill_mode="constant",
            fill_value=fill_value,
            seed=1,
        )
        xs = layer(xs)

        # Some pixels should be replaced with fill value
        self.assertTrue(tf.math.reduce_any(xs[0] == fill_value))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == fill_value))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_in_tf_function(self):
        xs = tf.cast(
            tf.stack([2 * tf.ones((100, 100, 1)), tf.ones((100, 100, 1))], axis=0),
            tf.float32,
        )

        patch_value = 0.0
        layer = preprocessing.RandomCutout(
            height_factor=0.5,
            width_factor=0.5,
            fill_mode="constant",
            fill_value=patch_value,
            seed=1,
        )

        @tf.function
        def augment(x):
            return layer(x)

        xs = augment(xs)

        # Some pixels should be replaced with fill value
        self.assertTrue(tf.math.reduce_any(xs[0] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))
