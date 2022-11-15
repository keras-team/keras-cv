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

from keras_cv.layers import TransformerEncoder


class TransformerEncoderTest(tf.test.TestCase):
    def test_return_type_and_shape(self):
        layer = TransformerEncoder(project_dim=128, num_heads=2, mlp_dim=128)

        inputs = tf.random.normal([1, 197, 128])
        output = layer(inputs, training=True)
        self.assertTrue(isinstance(output, tf.Tensor))
        self.assertLen(output, 1)
        self.assertEquals(output.shape, [1, 197, 128])

    def test_wrong_input_dims(self):
        layer = TransformerEncoder(project_dim=128, num_heads=2, mlp_dim=128)
        # Input dims must equal output dims because of the addition
        # of the residual to the final layer
        inputs = tf.random.normal([1, 197, 256])
        with self.assertRaisesRegexp(
            ValueError,
            "The input and output dimensionality must be the same, but the TransformerEncoder was provided with 256 and 128",
        ):
            layer(inputs, training=True)

    def test_wrong_project_dims(self):
        layer = TransformerEncoder(project_dim=256, num_heads=2, mlp_dim=128)
        # Input dims must equal output dims because of the addition
        # of the residual to the final layer
        inputs = tf.random.normal([1, 197, 128])
        with self.assertRaisesRegexp(
            ValueError,
            "The input and output dimensionality must be the same, but the TransformerEncoder was provided with 128 and 256",
        ):
            layer(inputs, training=True)
