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
import pytest
import tensorflow as tf

from keras_cv.layers.mbconv import MBConvBlock


class MBConvTest(tf.test.TestCase):
    @pytest.fixture(autouse=True)
    def cleanup_global_session(self):
        # Code before yield runs before the test
        tf.config.set_soft_device_placement(False)
        yield
        # Reset soft device placement to not interfere with other unit test files
        tf.config.set_soft_device_placement(True)
        tf.keras.backend.clear_session()

    def test_same_input_output_shapes(self):
        inputs = tf.random.normal(shape=(1, 64, 64, 32), dtype=tf.float32)
        layer = MBConvBlock(input_filters=32, output_filters=32)

        output = layer(inputs)
        self.assertEquals(output.shape, [1, 64, 64, 32])
        self.assertLen(output, 1)
        self.assertTrue(isinstance(output, tf.Tensor))

    def test_different_input_output_shapes(self):
        inputs = tf.random.normal(shape=(1, 64, 64, 32), dtype=tf.float32)
        layer = MBConvBlock(input_filters=32, output_filters=48)

        output = layer(inputs)
        self.assertEquals(output.shape, [1, 64, 64, 48])
        self.assertLen(output, 1)
        self.assertTrue(isinstance(output, tf.Tensor))

    def test_squeeze_excitation_ratio(self):
        inputs = tf.random.normal(shape=(1, 64, 64, 32), dtype=tf.float32)
        layer = MBConvBlock(input_filters=32, output_filters=48, se_ratio=0.25)

        output = layer(inputs)
        self.assertEquals(output.shape, [1, 64, 64, 48])
        self.assertLen(output, 1)
        self.assertTrue(isinstance(output, tf.Tensor))
