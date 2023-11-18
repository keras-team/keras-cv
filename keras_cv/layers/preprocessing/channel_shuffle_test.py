# Copyright 2023 The KerasCV Authors
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

from keras_cv.layers.preprocessing.channel_shuffle import ChannelShuffle
from keras_cv.tests.test_case import TestCase


class ChannelShuffleTest(TestCase):
    def test_return_shapes(self):
        xs = tf.ones((2, 512, 512, 3))

        layer = ChannelShuffle(groups=3)
        xs = layer(xs, training=True)
        self.assertEqual(xs.shape, [2, 512, 512, 3])

    def test_channel_shuffle_call_results_one_channel(self):
        xs = tf.cast(
            tf.stack(
                [3 * tf.ones((40, 40, 1)), 2 * tf.ones((40, 40, 1))],
                axis=0,
            ),
            dtype=tf.float32,
        )

        layer = ChannelShuffle(groups=1)
        xs = layer(xs, training=True)
        self.assertTrue(tf.math.reduce_any(xs[0] == 3.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == 2.0))

    def test_channel_shuffle_call_results_multi_channel(self):
        xs = tf.cast(
            tf.stack(
                [3 * tf.ones((40, 40, 20)), 2 * tf.ones((40, 40, 20))],
                axis=0,
            ),
            dtype=tf.float32,
        )

        layer = ChannelShuffle(groups=5)
        xs = layer(xs, training=True)
        self.assertTrue(tf.math.reduce_any(xs[0] == 3.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == 2.0))

    def test_non_square_image(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((1024, 512, 1)), tf.ones((1024, 512, 1))],
                axis=0,
            ),
            dtype=tf.float32,
        )

        layer = ChannelShuffle(groups=1)
        xs = layer(xs, training=True)
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_in_tf_function(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((100, 100, 1)), tf.ones((100, 100, 1))], axis=0
            ),
            dtype=tf.float32,
        )

        layer = ChannelShuffle(groups=1)

        @tf.function
        def augment(x):
            return layer(x, training=True)

        xs = augment(xs)
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_in_single_image(self):
        xs = tf.cast(
            tf.ones((512, 512, 1)),
            dtype=tf.float32,
        )

        layer = ChannelShuffle(groups=1)
        xs = layer(xs, training=True)
        self.assertTrue(tf.math.reduce_any(xs == 1.0))

    def test_channel_shuffle_on_batched_images_independently(self):
        image = tf.random.uniform((100, 100, 3))
        batched_images = tf.stack((image, image), axis=0)
        layer = ChannelShuffle(groups=3)

        results = layer(batched_images)

        self.assertNotAllClose(results[0], results[1])

    def test_config_with_custom_name(self):
        layer = ChannelShuffle(name="image_preproc")
        config = layer.get_config()
        layer_1 = ChannelShuffle.from_config(config)
        self.assertEqual(layer_1.name, layer.name)

    def test_output_dtypes(self):
        inputs = tf.constant([[[1], [2]], [[3], [4]]], dtype="float64")
        layer = ChannelShuffle(groups=1)
        self.assertAllEqual(layer(inputs).dtype, "float32")
        layer = ChannelShuffle(groups=1, dtype="uint8")
        self.assertAllEqual(layer(inputs).dtype, "uint8")

    def test_config(self):
        layer = ChannelShuffle(groups=5)
        config = layer.get_config()
        self.assertEqual(config["groups"], 5)
