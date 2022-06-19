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

from keras_cv.layers import SqueezeAndExciteBlock2D


class SqueezeAndExciteBlock2DTest(tf.test.TestCase):
    def maintains_shape(self):
        input_shape = (1, 4, 4, 8)
        inputs = tf.random.uniform(input_shape)

        layer = SqueezeAndExciteBlock2D(8, ratio=0.25)
        outputs = layer(inputs)
        self.assertEquals(inputs.shape, outputs.shape)

    def raises_invalid_ratio_error(self):
        with self.assertRaisesRegex(
                ValueError, "`ratio` should be a float"
                "between 0 and 1. Got (.*?)"):
            layer = SqueezeAndExciteBlock2D(8, ratio=1)

    def raises_invalid_ratio_error(self):
        with self.assertRaisesRegex(
                ValueError, "`filters` should be a positive"
                " integer. Got (.*?)"):
            layer = SqueezeAndExciteBlock2D(-8.7)
