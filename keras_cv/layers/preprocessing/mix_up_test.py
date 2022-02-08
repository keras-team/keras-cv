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

from keras_cv.layers.preprocessing.mix_up import MixUp

NUM_CLASSES = 10


class MixUpTest(tf.test.TestCase):
    def test_return_shapes(self):
        xs = tf.ones((2, 512, 512, 3))
        # randomly sample labels
        ys = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 2)
        ys = tf.squeeze(ys)
        ys = tf.one_hot(ys, NUM_CLASSES)

        layer = MixUp()
        xs, ys = layer(xs, ys)

        self.assertEqual(xs.shape, [2, 512, 512, 3])
        self.assertEqual(ys.shape, [2, 10])

    def test_mix_up_call_results(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((4, 4, 3)), tf.ones((4, 4, 3))],
                axis=0,
            ),
            tf.float32,
        )
        ys = tf.one_hot(tf.constant([0, 1]), 2)

        layer = MixUp()
        xs, ys = layer(xs, ys)

        # None of the individual values should still be close to 1 or 0
        self.assertNotAllClose(xs, 1.0)
        self.assertNotAllClose(xs, 2.0)

        # No labels should still be close to their originals
        self.assertNotAllClose(ys, 1.0)
        self.assertNotAllClose(ys, 0.0)

    def test_in_tf_function(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((4, 4, 3)), tf.ones((4, 4, 3))],
                axis=0,
            ),
            tf.float32,
        )
        ys = tf.one_hot(tf.constant([0, 1]), 2)

        layer = MixUp()

        @tf.function
        def augment(x, y):
            return layer(x, y)

        xs, ys = augment(xs, ys)

        # None of the individual values should still be close to 1 or 0
        self.assertNotAllClose(xs, 1.0)
        self.assertNotAllClose(xs, 2.0)

        # No labels should still be close to their originals
        self.assertNotAllClose(ys, 1.0)
        self.assertNotAllClose(ys, 0.0)
