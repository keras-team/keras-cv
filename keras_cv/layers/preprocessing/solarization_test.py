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

from keras_cv.layers.preprocessing.solarization import Solarization


class SolarizationTest(tf.test.TestCase):
    def test_range_0_to_255(self):
        solarization = Solarization()

        test_parameters = [
            {"input_value": 0, "expected_output_value": 255},
            {"input_value": 64, "expected_output_value": 191},
            {"input_value": 127, "expected_output_value": 128},
            {"input_value": 191, "expected_output_value": 64},
            {"input_value": 255, "expected_output_value": 0},
        ]

        for parameters in test_parameters:
            self._test_input_output(
                layer=solarization,
                input_value=parameters["input_value"],
                expected_value=parameters["expected_output_value"],
                dtype=tf.uint8,
            )

    def test_solarization_with_addition(self):
        solarization = Solarization(addition=10.0)

        test_parameters = [
            {"input_value": 0, "expected_output_value": 245},
            {"input_value": 255, "expected_output_value": 0},
        ]

        for parameters in test_parameters:
            self._test_input_output(
                layer=solarization,
                input_value=parameters["input_value"],
                expected_value=parameters["expected_output_value"],
                dtype=tf.float32,
            )

    def test_only_values_above_threshold_are_solarized_if_threshold_specified(self):
        solarization = Solarization(threshold=128)

        test_parameters = [
            {"input_value": 0, "expected_output_value": 0},
            {"input_value": 64, "expected_output_value": 64},
            {"input_value": 127, "expected_output_value": 127},
            {"input_value": 191, "expected_output_value": 64},
            {"input_value": 255, "expected_output_value": 0},
        ]

        for parameters in test_parameters:
            self._test_input_output(
                layer=solarization,
                input_value=parameters["input_value"],
                expected_value=parameters["expected_output_value"],
                dtype=tf.uint8,
            )

    def _test_input_output(self, layer, input_value, expected_value, dtype):
        dummy_input = tf.ones(shape=(2, 224, 224, 3), dtype=dtype) * input_value
        expected_output = tf.clip_by_value(
            (
                tf.ones(shape=(2, 224, 224, 3), dtype=layer.compute_dtype)
                * expected_value
            ),
            0,
            255,
        )

        output = layer(dummy_input)

        self.assertAllClose(output, expected_output)
