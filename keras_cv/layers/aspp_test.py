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

from keras_cv.layers import ASPP

class ASPPTest(tf.test.TestCase):

    def test_return_type_and_shape(self):
        layer = ASPP(level=4, dilation_rates=[6, 12, 18])
        c2 = tf.ones([2, 64, 64, 3])
        c3 = tf.ones([2, 32, 32, 3])
        c4 = tf.ones([2, 16, 16, 3])
        c5 = tf.ones([2, 8, 8, 3])

        inputs = {2: c2, 3: c3, 4: c4, 5: c5}
        output = layer(inputs)
        self.assertTrue(isinstance(output, dict))
        self.assertLen(output, 1)
        self.assertEquals(output[4].shape, [2, 16, 16, 256])

    def test_with_keras_tensor(self):
        layer = ASPP(level=4, dilation_rates=[6, 12, 18])
        c2 = tf.keras.layers.Input([64, 64, 3])
        c3 = tf.keras.layers.Input([32, 32, 3])
        c4 = tf.keras.layers.Input([16, 16, 3])
        c5 = tf.keras.layers.Input([8, 8, 3])

        inputs = {2: c2, 3: c3, 4: c4, 5: c5}
        output = layer(inputs)
        self.assertTrue(isinstance(output, dict))
        self.assertLen(output, 1)
        self.assertEquals(output[4].shape, [None, 16, 16, 256])

    def test_invalid_input_type(self):
        layer = ASPP(level=4, dilation_rates=[6, 12, 18])
        c4 = tf.keras.layers.Input([16, 16, 3])

        with self.assertRaisesRegexp(
                ValueError, "Expect the inputs to be a dict with int keys"):
            layer(c4, training=True)

