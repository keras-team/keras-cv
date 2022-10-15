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

from convolution_block_attention_module import CBAM


class CBAMTest(tf.test.TestCase):
    def test_maintains_shape(self):
        input_shape = (1, 128, 128, 32)
        inputs = tf.random.uniform(input_shape)

        layer = CBAM(32, ratio=0.25)
        outputs = layer(inputs)
        self.assertEquals(inputs.shape, outputs.shape)

    def test_custom_activation(self):
        def custom_activation(x):
            return x * tf.random.uniform(x.shape, seed=42)

        input_shape = (1, 128, 128, 32)
        inputs = tf.random.uniform(input_shape)

        layer = CBAM(
            32,
            ratio=0.25,
            channel_activation=custom_activation,
            spatial_activation=custom_activation,
        )
        outputs = layer(inputs)
        self.assertEquals(inputs.shape, outputs.shape)

    def test_raises_invalid_ratio_error(self):
        with self.assertRaisesRegex(
            ValueError, "`ratio` should be a float" " between 0 and 1. Got (.*?)"
        ):
            _ = CBAM(32, ratio=1.1)

    def test_raises_invalid_filters_error(self):
        with self.assertRaisesRegex(
            ValueError, "`filters` should be a positive" " integer. Got (.*?)"
        ):
            _ = CBAM(-32.7)
