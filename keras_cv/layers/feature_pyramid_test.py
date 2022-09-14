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

from keras_cv.layers import FeaturePyramid

class FeaturePyramidTest(tf.test.TestCase):
    def test_return_type_list(self):
        layer = FeaturePyramid(pyramid_levels=[2, 3, 4, 5])
        c2 = tf.keras.layers.Input([64, 64, 3])
        c3 = tf.keras.layers.Input([32, 32, 3])
        c4 = tf.keras.layers.Input([16, 16, 3])
        c5 = tf.keras.layers.Input([8, 8, 3])

        output = layer([c2, c3, c4, c5])
        self.assertTrue(isinstance(output, list))
        self.assertLen(output, 4)

    def test_return_type_dict(self):
        layer = FeaturePyramid(pyramid_levels=[2, 3, 4, 5])
        c2 = tf.keras.layers.Input([64, 64, 3])
        c3 = tf.keras.layers.Input([32, 32, 3])
        c4 = tf.keras.layers.Input([16, 16, 3])
        c5 = tf.keras.layers.Input([8, 8, 3])

        inputs = {'C2': c2,
                  'C3': c3,
                  'C4': c4,
                  'C5': c5}

        output = layer(inputs)
        self.assertTrue(isinstance(output, dict))
        self.assertEquals(output.keys(), ['P2', 'P3', 'P4', 'P5'])
