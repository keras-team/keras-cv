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

from keras_cv.layers.regularization.stochastic_depth import StochasticDepth


class StochasticDepthTest(tf.test.TestCase):
    FEATURE_SHAPE = (1, 14, 14, 256)

    def test_inputs_have_two_elements(self):
        inputs = tf.random.uniform(self.FEATURE_SHAPE, 0, 1)
        inputs = [inputs, inputs, inputs]

        with self.assertRaisesRegex(
            ValueError, "Input must be a list of length 2. " "Got input with length=3."
        ):
            StochasticDepth()(inputs)

    def test_eval_mode(self):
        inputs = tf.random.uniform(self.FEATURE_SHAPE, 0, 1)
        inputs = [inputs, inputs]

        rate = 0.5

        outputs = StochasticDepth(rate=rate)(inputs, training=False)

        self.assertAllClose(inputs[0] * (1 + rate), outputs)

    def test_training_mode(self):
        inputs = tf.random.uniform(self.FEATURE_SHAPE, 0, 1)
        inputs = [inputs, inputs]

        rate = 0.5

        outputs = StochasticDepth(rate=rate)(inputs, training=True)

        outputs_sum = tf.math.reduce_sum(outputs)
        inputs_sum = tf.math.reduce_sum(inputs[0])

        self.assertIn(outputs_sum, [inputs_sum, inputs_sum * 2])
