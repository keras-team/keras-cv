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
    def test_return_type_dict(self):
        layer = FeaturePyramid(min_level=2, max_level=5)
        c2 = tf.ones([2, 64, 64, 3])
        c3 = tf.ones([2, 32, 32, 3])
        c4 = tf.ones([2, 16, 16, 3])
        c5 = tf.ones([2, 8, 8, 3])

        inputs = {2: c2, 3: c3, 4: c4, 5: c5}
        output = layer(inputs)
        self.assertTrue(isinstance(output, dict))
        self.assertEquals(sorted(output.keys()), [2, 3, 4, 5])

    def test_result_shapes(self):
        layer = FeaturePyramid(min_level=2, max_level=5)
        c2 = tf.ones([2, 64, 64, 3])
        c3 = tf.ones([2, 32, 32, 3])
        c4 = tf.ones([2, 16, 16, 3])
        c5 = tf.ones([2, 8, 8, 3])

        inputs = {2: c2, 3: c3, 4: c4, 5: c5}
        output = layer(inputs)
        for level in inputs.keys():
            self.assertEquals(output[level].shape[1], inputs[level].shape[1])
            self.assertEquals(output[level].shape[2], inputs[level].shape[2])
            self.assertEquals(output[level].shape[3], layer.num_channels)

        # Test with different resolution and channel size
        c2 = tf.ones([2, 64, 128, 4])
        c3 = tf.ones([2, 32, 64, 8])
        c4 = tf.ones([2, 16, 32, 16])
        c5 = tf.ones([2, 8, 16, 32])

        inputs = {2: c2, 3: c3, 4: c4, 5: c5}
        layer = FeaturePyramid(min_level=2, max_level=5)
        output = layer(inputs)
        for level in inputs.keys():
            self.assertEquals(output[level].shape[1], inputs[level].shape[1])
            self.assertEquals(output[level].shape[2], inputs[level].shape[2])
            self.assertEquals(output[level].shape[3], layer.num_channels)

    def test_with_keras_input_tensor(self):
        # This mimic the model building with Backbone network
        layer = FeaturePyramid(min_level=2, max_level=5)
        c2 = tf.keras.layers.Input([64, 64, 3])
        c3 = tf.keras.layers.Input([32, 32, 3])
        c4 = tf.keras.layers.Input([16, 16, 3])
        c5 = tf.keras.layers.Input([8, 8, 3])

        inputs = {2: c2, 3: c3, 4: c4, 5: c5}
        output = layer(inputs)
        for level in inputs.keys():
            self.assertEquals(output[level].shape[1], inputs[level].shape[1])
            self.assertEquals(output[level].shape[2], inputs[level].shape[2])
            self.assertEquals(output[level].shape[3], layer.num_channels)

    def test_invalid_lateral_layers(self):
        lateral_layers = [tf.keras.layers.Conv2D(256, 1)] * 3
        with self.assertRaisesRegexp(ValueError, "Expect lateral_layers to be a dict"):
            _ = FeaturePyramid(min_level=2, max_level=5, lateral_layers=lateral_layers)
        lateral_layers = {
            2: tf.keras.layers.Conv2D(256, 1),
            3: tf.keras.layers.Conv2D(256, 1),
            4: tf.keras.layers.Conv2D(256, 1),
        }
        with self.assertRaisesRegexp(ValueError, "with keys as .* [2, 3, 4, 5]"):
            _ = FeaturePyramid(min_level=2, max_level=5, lateral_layers=lateral_layers)

    def test_invalid_output_layers(self):
        output_layers = [tf.keras.layers.Conv2D(256, 3)] * 3
        with self.assertRaisesRegexp(ValueError, "Expect output_layers to be a dict"):
            _ = FeaturePyramid(min_level=2, max_level=5, output_layers=output_layers)
        output_layers = {
            2: tf.keras.layers.Conv2D(256, 3),
            3: tf.keras.layers.Conv2D(256, 3),
            4: tf.keras.layers.Conv2D(256, 3),
        }
        with self.assertRaisesRegexp(ValueError, "with keys as .* [2, 3, 4, 5]"):
            _ = FeaturePyramid(min_level=2, max_level=5, output_layers=output_layers)

    def test_invalid_input_features(self):
        layer = FeaturePyramid(min_level=2, max_level=5)

        c2 = tf.ones([2, 64, 64, 3])
        c3 = tf.ones([2, 32, 32, 3])
        c4 = tf.ones([2, 16, 16, 3])
        c5 = tf.ones([2, 8, 8, 3])
        list_input = [c2, c3, c4, c5]
        with self.assertRaisesRegexp(ValueError, "expects input features to be a dict"):
            layer(list_input)

        dict_input_with_missing_feature = {2: c2, 3: c3, 4: c4}
        with self.assertRaisesRegexp(ValueError, "Expect feature keys.*[2, 3, 4, 5]"):
            layer(dict_input_with_missing_feature)
