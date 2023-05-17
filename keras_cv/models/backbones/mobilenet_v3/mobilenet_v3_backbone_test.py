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

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_cv.models.backbones.mobilenet_v3.mobilenet_v3_backbone import (
    MobileNetV3Backbone,
)
from keras_cv.models.backbones.mobilenet_v3.mobilenet_v3_backbone import (
    MobileNetV3SmallBackbone,
)
from keras_cv.utils.train import get_feature_extractor

# from https://arxiv.org/pdf/1905.02244.pdf
pyramid_level_input_shapes = {
    "mobilenet_v3_small": {
        3: [None, 28, 28, 24],
        4: [None, 14, 14, 48],
        5: [None, 7, 7, 96]
    },
    "mobilenet_v3_large": {
        3: [None, 28, 28, 40],
        4: [None, 14, 14, 112],
        5: [None, 7, 7, 160]
    }
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

    @parameterized.named_parameters(
        ("small", "mobilenet_v3_small"),
        ("large", "mobilenet_v3_large"),
    )
    def test_create_backbone_model_with_level_config(self, preset):
        metadata = MobileNetV3Backbone.presets[preset]
        metadata["config"]["input_shape"] = [224, 224, 3]
        model = MobileNetV3Backbone.from_config(metadata["config"])

        levels = [3, 4, 5]
        layer_names = [model.pyramid_level_inputs[level] for level in levels]
        backbone_model = get_feature_extractor(model, layer_names, levels)
        inputs = tf.keras.Input(shape=[224, 224, 3])
        outputs = backbone_model(inputs)

        # confirm the shapes of the pyramid level input
        self.assertLen(outputs, len(levels))
        self.assertEquals(list(outputs.keys()), [3, 4, 5])
        for level in levels:
            self.assertEquals(outputs[level].shape,
                              pyramid_level_input_shapes[preset][level])

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
