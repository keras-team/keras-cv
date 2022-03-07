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

from keras_cv.layers.regularization.dropblock_2d import DropBlock2D


class DropBlock2DTest(tf.test.TestCase):
    FEATURE_SHAPE = (1, 14, 14, 256)  # Shape of ResNet block group 3
    rng = tf.random.Generator.from_non_deterministic_state()

    def test_layer_not_created_with_invalid_block_size(self):
        with self.assertRaises(ValueError):
            DropBlock2D(dropblock_size=0)

    def test_layer_not_created_with_invalid_keep_probability(self):
        for keep_probability in [1.1, -0.1]:
            with self.assertRaises(ValueError):
                DropBlock2D(keep_probability=keep_probability)

    def test_input_unchanged_in_eval_mode(self):
        dummy_inputs = self.rng.uniform(shape=self.FEATURE_SHAPE)
        layer = DropBlock2D()

        output = layer(dummy_inputs, training=False)

        tf.debugging.assert_near(dummy_inputs, output)

    def test_input_unchanged_with_keep_probability_equal_to_one(self):
        dummy_inputs = self.rng.uniform(shape=self.FEATURE_SHAPE)
        layer = DropBlock2D(keep_probability=1.0)

        output = layer(dummy_inputs, training=True)

        tf.debugging.assert_near(dummy_inputs, output)

    def test_input_gets_partially_zeroed_out_in_train_mode(self):
        dummy_inputs = self.rng.uniform(shape=self.FEATURE_SHAPE)
        layer = DropBlock2D()

        output = layer(dummy_inputs, training=True)
        num_input_zeros = self._count_zeros(dummy_inputs)
        num_output_zeros = self._count_zeros(output)

        self.assertGreater(num_output_zeros, num_input_zeros)

    def test_batched_input_gets_partially_zeroed_out_in_train_mode(self):
        batched_shape = (4, *self.FEATURE_SHAPE[1:])
        dummy_inputs = self.rng.uniform(shape=batched_shape)
        layer = DropBlock2D()

        output = layer(dummy_inputs, training=True)
        num_input_zeros = self._count_zeros(dummy_inputs)
        num_output_zeros = self._count_zeros(output)

        self.assertGreater(num_output_zeros, num_input_zeros)

    @staticmethod
    def _count_zeros(tensor: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(tf.cast(tensor == 0, dtype=tf.int32))

    def test_works_with_xla(self):
        dummy_inputs = self.rng.uniform(shape=self.FEATURE_SHAPE)
        layer = DropBlock2D()

        @tf.function(jit_compile=True)
        def apply(x):
            return layer(x, training=True)

        layer(dummy_inputs)
