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
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_cv.layers.preprocessing.five_crop import FiveCrop


class FiveCropTest(tf.test.TestCase):

    def test_output_shape(self):
        orig_height = 300
        orig_width = 300
        channels = 3
        crop_size = 50
        input = tf.random.uniform(
            shape=[orig_height, orig_width, channels],
        )

        layer = FiveCrop((crop_size, crop_size))
        tl_actual_output, tr_actual_output, bl_actual_output , br_actual_output, center_actual_output = layer(input)

        expected_output = tf.random.uniform(
            shape=(
                crop_size,
                crop_size,
                channels,
            ),
        )

        self.assertAllEqual(expected_output.shape, tl_actual_output.shape)
        self.assertAllEqual(expected_output.shape, tr_actual_output.shape)
        self.assertAllEqual(expected_output.shape, bl_actual_output.shape)
        self.assertAllEqual(expected_output.shape, br_actual_output.shape)
        self.assertAllEqual(expected_output.shape, center_actual_output.shape)

    def test_batched_input(self):
        num_samples = 2
        orig_height = 300
        orig_width = 300
        channels = 3
        crop_size = 50
        input = tf.random.uniform(
            shape=[num_samples, orig_height, orig_width, channels],
        )

        layer = FiveCrop((crop_size, crop_size))
        tl_actual_output, tr_actual_output, bl_actual_output , br_actual_output, center_actual_output = layer(input)

        expected_output = tf.random.uniform(
            shape=(
                num_samples,
                crop_size,
                crop_size,
                channels,
            ),
        )

        self.assertAllEqual(expected_output.shape, tl_actual_output.shape)
        self.assertAllEqual(expected_output.shape, tr_actual_output.shape)
        self.assertAllEqual(expected_output.shape, bl_actual_output.shape)
        self.assertAllEqual(expected_output.shape, br_actual_output.shape)
        self.assertAllEqual(expected_output.shape, center_actual_output.shape)

    def test_config_with_custom_name(self):
        layer = FiveCrop((50,50), name="Five Crop")
        config = layer.get_config()
        layer_1 = FiveCrop.from_config(config)
        self.assertEqual(layer_1.name, layer.name)
    
    def test_out_of_bounds_crop(self):

        orig_height = 300
        orig_width = 300
        channels = 3
        crop_size = 50
        input = tf.random.uniform(
            shape=[orig_height, orig_width, channels],
        )

        layer = FiveCrop((350, 350))

        with pytest.raises(Exception):
            layer(input)
