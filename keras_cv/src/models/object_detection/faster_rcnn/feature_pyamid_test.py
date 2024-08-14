# Copyright 2024 The KerasCV Authors
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

from keras_cv.src.backend import keras
from keras_cv.src.models.object_detection.faster_rcnn import FeaturePyramid
from keras_cv.src.tests.test_case import TestCase


class FeaturePyramidTest(TestCase):
    def test_return_type_dict(self):
        layer = FeaturePyramid(min_level=2, max_level=5)
        c2 = np.ones([2, 64, 64, 3])
        c3 = np.ones([2, 32, 32, 3])
        c4 = np.ones([2, 16, 16, 3])
        c5 = np.ones([2, 8, 8, 3])

        inputs = {"P2": c2, "P3": c3, "P4": c4, "P5": c5}
        output = layer(inputs)
        self.assertTrue(isinstance(output, dict))
        self.assertEquals(sorted(output.keys()), ["P2", "P3", "P4", "P5", "P6"])

    def test_result_shapes(self):
        layer = FeaturePyramid(min_level=2, max_level=5)
        c2 = np.ones([2, 64, 64, 3])
        c3 = np.ones([2, 32, 32, 3])
        c4 = np.ones([2, 16, 16, 3])
        c5 = np.ones([2, 8, 8, 3])

        inputs = {"P2": c2, "P3": c3, "P4": c4, "P5": c5}
        output = layer(inputs)
        for level in inputs.keys():
            self.assertEquals(output[level].shape[1], inputs[level].shape[1])
            self.assertEquals(output[level].shape[2], inputs[level].shape[2])
            self.assertEquals(output[level].shape[3], layer.num_channels)

        # Test with different resolution and channel size
        c2 = np.ones([2, 64, 128, 4])
        c3 = np.ones([2, 32, 64, 8])
        c4 = np.ones([2, 16, 32, 16])
        c5 = np.ones([2, 8, 16, 32])

        inputs = {"P2": c2, "P3": c3, "P4": c4, "P5": c5}
        layer = FeaturePyramid(min_level=2, max_level=5)
        output = layer(inputs)
        for level in inputs.keys():
            self.assertEquals(output[level].shape[1], inputs[level].shape[1])
            self.assertEquals(output[level].shape[2], inputs[level].shape[2])
            self.assertEquals(output[level].shape[3], layer.num_channels)

    def test_with_keras_input_tensor(self):
        # This mimic the model building with Backbone network
        layer = FeaturePyramid(min_level=2, max_level=5)
        c2 = keras.layers.Input([64, 64, 3])
        c3 = keras.layers.Input([32, 32, 3])
        c4 = keras.layers.Input([16, 16, 3])
        c5 = keras.layers.Input([8, 8, 3])

        inputs = {"P2": c2, "P3": c3, "P4": c4, "P5": c5}
        output = layer(inputs)
        for level in inputs.keys():
            self.assertEquals(output[level].shape[1], inputs[level].shape[1])
            self.assertEquals(output[level].shape[2], inputs[level].shape[2])
            self.assertEquals(output[level].shape[3], layer.num_channels)

    def test_invalid_lateral_layers(self):
        lateral_layers = [keras.layers.Conv2D(256, 1)] * 3
        with self.assertRaisesRegexp(
            ValueError, "Expect lateral_layers to be a dict"
        ):
            _ = FeaturePyramid(
                min_level=2, max_level=5, lateral_layers=lateral_layers
            )
        lateral_layers = {
            "P2": keras.layers.Conv2D(256, 1),
            "P3": keras.layers.Conv2D(256, 1),
            "P4": keras.layers.Conv2D(256, 1),
        }
        with self.assertRaisesRegexp(
            ValueError, "with keys as .* ['P2', 'P3', 'P4', 'P5']"
        ):
            _ = FeaturePyramid(
                min_level=2, max_level=5, lateral_layers=lateral_layers
            )

    def test_invalid_output_layers(self):
        output_layers = [keras.layers.Conv2D(256, 3)] * 3
        with self.assertRaisesRegexp(
            ValueError, "Expect output_layers to be a dict"
        ):
            _ = FeaturePyramid(
                min_level=2, max_level=5, output_layers=output_layers
            )
        output_layers = {
            "P2": keras.layers.Conv2D(256, 3),
            "P3": keras.layers.Conv2D(256, 3),
            "P4": keras.layers.Conv2D(256, 3),
        }
        with self.assertRaisesRegexp(
            ValueError, "with keys as .* ['P2', 'P3', 'P4', 'P5']"
        ):
            _ = FeaturePyramid(
                min_level=2, max_level=5, output_layers=output_layers
            )

    def test_invalid_input_features(self):
        layer = FeaturePyramid(min_level=2, max_level=5)

        c2 = np.ones([2, 64, 64, 3])
        c3 = np.ones([2, 32, 32, 3])
        c4 = np.ones([2, 16, 16, 3])
        c5 = np.ones([2, 8, 8, 3])
        inputs = {"P2": c2, "P3": c3, "P4": c4, "P5": c5}
        # Build required for Keas 3
        _ = layer(inputs)
        list_input = [c2, c3, c4, c5]
        with self.assertRaisesRegexp(
            ValueError, "expects input features to be a dict"
        ):
            layer(list_input)

        dict_input_with_missing_feature = {"P2": c2, "P3": c3, "P4": c4}
        with self.assertRaisesRegexp(
            ValueError, "Expect feature keys.*['P2', 'P3', 'P4', 'P5']"
        ):
            layer(dict_input_with_missing_feature)
