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

from keras_cv.layers.preprocessing.posterization import Posterization


class PosterizationTest(tf.test.TestCase):
    rng = tf.random.Generator.from_seed(1234)

    def test_raises_error_on_invalid_bits_parameter(self):
        invalid_values = [-1, 0, 9, 24]

        for value in invalid_values:
            with self.assertRaises(AssertionError):
                Posterization(bits=value)

    def test_output_shape_unchanged(self):
        dummy_input = self.rng.uniform(shape=(2, 224, 224, 3), minval=0, maxval=256)
        layer = Posterization(bits=4)

        output = layer(dummy_input)

        self.assertEqual(output.shape, dummy_input.shape)

    def test_output_dtype_unchanged(self):
        dtypes = [tf.float32, tf.int32, tf.uint8]
        dummy_input = self.rng.uniform(shape=(2, 224, 224, 3), minval=0, maxval=256)
        layer = Posterization(bits=4)

        for dtype in dtypes:
            inputs = tf.cast(dummy_input, dtype)
            output = layer(inputs)
            self.assertEqual(output.dtype, dtype)
