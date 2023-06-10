# Copyright 2023 The KerasCV Authors
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

import os

import pytest
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_cv.models.backbones.mobilenet_v3.mobilenet_v3_aliases import (
    MobileNetV3SmallBackbone,
)
from keras_cv.models.backbones.mobilenet_v3.mobilenet_v3_backbone import (
    MobileNetV3Backbone,
)
from keras_cv.utils.train import get_feature_extractor

# from https://arxiv.org/pdf/1905.02244.pdf
pyramid_level_input_shapes = {
    "mobilenet_v3_small": {
        "P3": [None, 28, 28, 24],
        "P4": [None, 14, 14, 48],
        "P5": [None, 7, 7, 96],
    },
    "mobilenet_v3_large": {
        "P3": [None, 28, 28, 40],
        "P4": [None, 14, 14, 112],
        "P5": [None, 7, 7, 160],
    },
}


class MobileNetV3BackboneTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.input_batch = tf.ones(shape=(2, 224, 224, 3))

    def test_valid_call(self):
        model = MobileNetV3SmallBackbone(
            include_rescaling=False,
        )
        model(self.input_batch)

    def test_valid_call_with_rescaling(self):
        model = MobileNetV3SmallBackbone(
            include_rescaling=True,
        )
        model(self.input_batch)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self, save_format, filename):
        model = MobileNetV3SmallBackbone()
        model_output = model(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(), filename)
        model.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, MobileNetV3Backbone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)

    def test_feature_pyramid_inputs(self):
        model = MobileNetV3SmallBackbone()
        backbone_model = get_feature_extractor(
            model,
            model.pyramid_level_inputs.values(),
            model.pyramid_level_inputs.keys(),
        )
        input_size = 256
        inputs = tf.keras.Input(shape=[input_size, input_size, 3])
        outputs = backbone_model(inputs)
        expected_levels = ["P1", "P2", "P3", "P4", "P5"]
        self.assertEquals(list(outputs.keys()), expected_levels)
        # Size for each feature map at Pn is represents a feature map 2^n
        # times smaller in width and height than the input image.
        for level in model.pyramid_level_inputs:
            level_int = int(level[1:])
            self.assertEquals(
                outputs[level].shape[1], input_size / 2**level_int
            )
            self.assertEquals(
                outputs[level].shape[2], input_size / 2**level_int
            )

    @parameterized.named_parameters(
        ("one_channel", 1),
        ("four_channels", 4),
    )
    def test_application_variable_input_channels(self, num_channels):
        model = MobileNetV3SmallBackbone(
            input_shape=(None, None, num_channels),
            include_rescaling=False,
        )
        self.assertEqual(model.output_shape, (None, None, None, 576))


if __name__ == "__main__":
    tf.test.main()
