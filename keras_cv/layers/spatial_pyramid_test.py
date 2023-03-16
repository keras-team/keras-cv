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

from keras_cv.layers import SpatialPyramidPooling


class SpatialPyramidPoolingTest(tf.test.TestCase):
    def test_return_type_and_shape(self):
        layer = SpatialPyramidPooling(dilation_rates=[6, 12, 18])
        c4 = tf.ones([2, 16, 16, 3])

        inputs = c4
        output = layer(inputs, training=True)
        self.assertEquals(output.shape, [2, 16, 16, 256])

    def test_with_keras_tensor(self):
        layer = SpatialPyramidPooling(dilation_rates=[6, 12, 18])
        c4 = tf.keras.layers.Input([16, 16, 3])

        inputs = c4
        output = layer(inputs, training=True)
        self.assertEquals(output.shape, [None, 16, 16, 256])
