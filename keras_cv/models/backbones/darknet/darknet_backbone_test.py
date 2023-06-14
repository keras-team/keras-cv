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

from keras_cv.models.backbones.darknet.darknet_backbone import DarkNet21Backbone
from keras_cv.models.backbones.darknet.darknet_backbone import DarkNet53Backbone
from keras_cv.models.backbones.darknet.darknet_backbone import DarkNetBackbone
from keras_cv.utils.train import get_feature_extractor


class DarkNetBackboneTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.input_batch = tf.ones(shape=(2, 224, 224, 3))

    def test_valid_call(self):
        model = DarkNetBackbone(
            stackwise_blocks=[2, 8, 8, 4],
            include_rescaling=False,
        )
        model(self.input_batch)

    def test_valid_call_applications_model(self):
        model = DarkNet53Backbone()
        model(self.input_batch)

    def test_valid_call_with_rescaling(self):
        model = DarkNetBackbone(
            stackwise_blocks=[2, 8, 8, 4],
            include_rescaling=True,
        )
        model(self.input_batch)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        model = DarkNetBackbone(
            stackwise_blocks=[2, 8, 8, 4],
            include_rescaling=False,
        )
        model_output = model(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(), filename)
        model.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, DarkNetBackbone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_alias_model(self, save_format, filename):
        model = DarkNet53Backbone()
        model_output = model(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(), filename)
        model.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        # Note that these aliases serialized as the base class
        self.assertIsInstance(restored_model, DarkNetBackbone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)

    def test_create_backbone_model_from_alias_model(self):
        model = DarkNet53Backbone(
            include_rescaling=False,
        )
        backbone_model = get_feature_extractor(
            model,
            model.pyramid_level_inputs.values(),
            model.pyramid_level_inputs.keys(),
        )
        inputs = tf.keras.Input(shape=[256, 256, 3])
        outputs = backbone_model(inputs)
        self.assertLen(outputs, 4)
        self.assertEquals(list(outputs.keys()), [2, 3, 4, 5])
        self.assertEquals(outputs[2].shape, [None, 64, 64, 256])
        self.assertEquals(outputs[3].shape, [None, 32, 32, 512])
        self.assertEquals(outputs[4].shape, [None, 16, 16, 1024])
        self.assertEquals(outputs[5].shape, [None, 8, 8, 2048])

    def test_create_backbone_model_with_level_config(self):
        model = DarkNetBackbone(
            stackwise_blocks=[2, 8, 8, 4],
            include_rescaling=False,
            input_shape=[256, 256, 3],
        )
        levels = [3, 4]
        layer_names = [model.pyramid_level_inputs[level] for level in [3, 4]]
        backbone_model = get_feature_extractor(model, layer_names, levels)
        inputs = tf.keras.Input(shape=[256, 256, 3])
        outputs = backbone_model(inputs)
        self.assertLen(outputs, 2)
        self.assertEquals(list(outputs.keys()), [3, 4])
        self.assertEquals(outputs[3].shape, [None, 128, 128, 64])
        self.assertEquals(outputs[4].shape, [None, 64, 64, 128])

    @parameterized.named_parameters(
        ("one_channel", 1),
        ("four_channels", 4),
    )
    def test_application_variable_input_channels(self, num_channels):
        model = DarkNetBackbone(
            stackwise_blocks=[2, 8, 8, 4],
            input_shape=(None, None, num_channels),
            include_rescaling=False,
        )
        self.assertEqual(model.output_shape, (None, None, None, 512))

    @parameterized.named_parameters(
        ("21", DarkNet21Backbone),
        ("53", DarkNet53Backbone),
    )
    def test_specific_arch_forward_pass(self, arch_class):
        backbone = arch_class()
        backbone(tf.random.uniform(shape=[2, 256, 256, 3]))


if __name__ == "__main__":
    tf.test.main()
