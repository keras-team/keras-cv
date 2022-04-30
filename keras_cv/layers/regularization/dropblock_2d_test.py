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
        invalid_sizes = [0, -10, (5, -2), (0, 7), (1, 2, 3, 4)]
        for size in invalid_sizes:
            with self.assertRaises(ValueError):
                DropBlock2D(block_size=size, rate=0.1)

    def test_layer_not_created_with_invalid_rate(self):
        invalid_rates = [1.1, -0.1]
        for rate in invalid_rates:
            with self.assertRaises(ValueError):
                DropBlock2D(rate=rate, block_size=7)

    def test_input_unchanged_in_eval_mode(self):
        dummy_inputs = self.rng.uniform(shape=self.FEATURE_SHAPE)
        layer = DropBlock2D(rate=0.1, block_size=7)

        output = layer(dummy_inputs, training=False)

        self.assertAllClose(dummy_inputs, output)

    def test_input_unchanged_with_rate_equal_to_zero(self):
        dummy_inputs = self.rng.uniform(shape=self.FEATURE_SHAPE)
        layer = DropBlock2D(rate=0.0, block_size=7)

        output = layer(dummy_inputs, training=True)

        self.assertAllClose(dummy_inputs, output)

    def test_input_gets_partially_zeroed_out_in_train_mode(self):
        dummy_inputs = self.rng.uniform(shape=self.FEATURE_SHAPE)
        layer = DropBlock2D(rate=0.1, block_size=7)

        output = layer(dummy_inputs, training=True)
        num_input_zeros = self._count_zeros(dummy_inputs)
        num_output_zeros = self._count_zeros(output)

        self.assertGreater(num_output_zeros, num_input_zeros)

    def test_batched_input_gets_partially_zeroed_out_in_train_mode(self):
        batched_shape = (4, *self.FEATURE_SHAPE[1:])
        dummy_inputs = self.rng.uniform(shape=batched_shape)
        layer = DropBlock2D(rate=0.1, block_size=7)

        output = layer(dummy_inputs, training=True)
        num_input_zeros = self._count_zeros(dummy_inputs)
        num_output_zeros = self._count_zeros(output)

        self.assertGreater(num_output_zeros, num_input_zeros)

    def test_input_gets_partially_zeroed_out_with_non_square_block_size(self):
        dummy_inputs = self.rng.uniform(shape=self.FEATURE_SHAPE)
        layer = DropBlock2D(rate=0.1, block_size=(7, 10))

        output = layer(dummy_inputs, training=True)
        num_input_zeros = self._count_zeros(dummy_inputs)
        num_output_zeros = self._count_zeros(output)

        self.assertGreater(num_output_zeros, num_input_zeros)

    @staticmethod
    def _count_zeros(tensor: tf.Tensor) -> tf.Tensor:
        return tf.size(tensor) - tf.math.count_nonzero(tensor, dtype=tf.int32)

    def test_works_with_xla(self):
        dummy_inputs = self.rng.uniform(shape=self.FEATURE_SHAPE)
        layer = DropBlock2D(rate=0.1, block_size=7)

        @tf.function(jit_compile=True)
        def apply(x):
            return layer(x, training=True)

        apply(dummy_inputs)
