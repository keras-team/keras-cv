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
from absl.testing import parameterized

from keras_cv.layers.preprocessing.solarization import Solarization


class SolarizationTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("0_255", 0, 255),
        ("64_191", 64, 191),
        ("127_128", 127, 128),
        ("191_64", 191, 64),
        ("255_0", 255, 0),
    )
    def test_output_values(self, input_value, expected_value):
        solarization = Solarization(value_range=(0, 255))

        self._test_input_output(
            layer=solarization,
            input_value=input_value,
            expected_value=expected_value,
            dtype=tf.uint8,
        )

    @parameterized.named_parameters(
        ("0_245", 0, 245),
        ("255_0", 255, 0),
    )
    def test_solarization_with_addition(self, input_value, output_value):
        solarization = Solarization(addition_factor=(10.0, 10.0), value_range=(0, 255))
        self._test_input_output(
            layer=solarization,
            input_value=input_value,
            expected_value=output_value,
            dtype=tf.float32,
        )

    @parameterized.named_parameters(
        ("0_0", 0, 0),
        ("64_64", 64, 64),
        ("127_127", 127, 127),
        ("191_64", 191, 64),
        ("255_0", 255, 0),
    )
    def test_only_values_above_threshold_are_solarized(self, input_value, output_value):
        solarization = Solarization(threshold_factor=(128, 128), value_range=(0, 255))

        self._test_input_output(
            layer=solarization,
            input_value=input_value,
            expected_value=output_value,
            dtype=tf.uint8,
        )

    def _test_input_output(self, layer, input_value, expected_value, dtype):
        input = tf.ones(shape=(2, 224, 224, 3), dtype=dtype) * input_value
        expected_output = tf.clip_by_value(
            (
                tf.ones(shape=(2, 224, 224, 3), dtype=layer.compute_dtype)
                * expected_value
            ),
            0,
            255,
        )

        output = layer(input)

        self.assertAllClose(output, expected_output)
