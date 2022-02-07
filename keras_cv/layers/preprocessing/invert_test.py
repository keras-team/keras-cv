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

from keras_cv.layers.preprocessing.invert import Invert


class InvertTest(tf.test.TestCase):
    invert = Invert()

    def test_return_shape_unchanged(self):
        dummy_input = tf.ones(shape=(2, 224, 224, 3))

        output = self.invert(dummy_input)

        self.assertEqual(dummy_input.shape, output.shape)

    def test_return_values(self):
        test_parameters = [
            {"input_value": 0, "expected_value": 255},
            {"input_value": 255, "expected_value": 0},
            {"input_value": 127.5, "expected_value": 127.5},
        ]

        for parameters in test_parameters:
            self._test_input_output(**parameters)

    def _test_input_output(self, input_value, expected_value):
        dummy_inputs = tf.ones(shape=(1, 224, 224, 3)) * input_value

        outputs = self.invert(dummy_inputs)

        self.assertTrue((outputs == expected_value).numpy().all())
