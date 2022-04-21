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

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from keras_cv.layers import preprocessing


class RandomColorJitterTest(tf.test.TestCase, parameterized.TestCase):
    # Test 1: Check input and output shape. It should match.
    def test_return_shapes(self):
        batch_input = tf.ones((2, 512, 512, 3))
        non_square_batch_input = tf.ones((2, 1024, 512, 3))
        unbatch_input = tf.ones((512, 512, 3))

        layer = preprocessing.RandomColorJitter(
            value_range=(0, 255),
            brightness_factor=0.5,
            contrast_factor=(0.5, 0.9),
            saturation_factor=(0.5, 0.9),
            hue_factor=0.5,
        )
        batch_output = layer(batch_input, training=True)
        non_square_batch_output = layer(non_square_batch_input, training=True)
        unbatch_output = layer(unbatch_input, training=True)

        self.assertEqual(batch_output.shape, [2, 512, 512, 3])
        self.assertEqual(non_square_batch_output.shape, [2, 1024, 512, 3])
        self.assertEqual(unbatch_output.shape, [512, 512, 3])

    # Test 2: Check if the factor ranges are set properly.
    def test_factor_range(self):
        layer = preprocessing.RandomColorJitter(
            value_range=(0, 255),
            brightness_factor=(-0.2, 0.5),
            contrast_factor=(0.5, 0.9),
            saturation_factor=(0.5, 0.9),
            hue_factor=(0.5, 0.9),
        )

        self.assertEqual(layer.brightness_factor, (-0.2, 0.5))
        self.assertEqual(layer.contrast_factor, (0.5, 0.9))
        self.assertEqual(layer.saturation_factor, (0.5, 0.9))
        self.assertEqual(layer.hue_factor, (0.5, 0.9))

    # Test 3: Test if it is OK to run on graph mode.
    def test_in_tf_function(self):
        inputs = tf.ones((2, 512, 512, 3))

        layer = preprocessing.RandomColorJitter(
            value_range=(0, 255),
            brightness_factor=0.5,
            contrast_factor=(0.5, 0.9),
            saturation_factor=(0.5, 0.9),
            hue_factor=0.5,
        )

        @tf.function
        def augment(x):
            return layer(x, training=True)

        outputs = augment(inputs)
        self.assertNotAllClose(inputs, outputs)

    # Test 4: Check if get_config and from_config work as expected.
    def test_config(self):
        layer = preprocessing.RandomColorJitter(
            value_range=(0, 255),
            brightness_factor=0.5,
            contrast_factor=(0.5, 0.9),
            saturation_factor=(0.5, 0.9),
            hue_factor=0.5,
        )

        config = layer.get_config()
        self.assertEqual(config["brightness_factor"], 0.5)
        self.assertEqual(config["contrast_factor"], (0.5, 0.9))
        self.assertEqual(config["saturation_factor"], (0.5, 0.9))
        self.assertEqual(config["hue_factor"], 0.5)

        reconstructed_layer = preprocessing.RandomColorJitter.from_config(config)
        self.assertEqual(reconstructed_layer.brightness_factor, layer.brightness_factor)
        self.assertEqual(reconstructed_layer.contrast_factor, layer.contrast_factor)
        self.assertEqual(reconstructed_layer.saturation_factor, layer.saturation_factor)
        self.assertEqual(reconstructed_layer.hue_factor, layer.hue_factor)

    # Test 5: Check if inference model is OK.
    def test_inference(self):
        layer = preprocessing.RandomColorJitter(
            value_range=(0, 255),
            brightness_factor=0.5,
            contrast_factor=(0.5, 0.9),
            saturation_factor=(0.5, 0.9),
            hue_factor=0.5,
        )
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output)
