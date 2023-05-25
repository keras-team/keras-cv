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

from keras_cv import use_keras_core

if use_keras_core():
    from keras_core import Input
    from keras_core.operations import ones
    from keras_core.saving import load_model
else:
    from tensorflow import ones
    from keras.models import load_model
    from keras import Input

import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_cv.models.backbones.resnet_v1.resnet_v1_backbone import (
    ResNet18Backbone,
)
from keras_cv.models.backbones.resnet_v1.resnet_v1_backbone import (
    ResNet50Backbone,
)
from keras_cv.models.backbones.resnet_v1.resnet_v1_backbone import (
    ResNet101Backbone,
)
from keras_cv.models.backbones.resnet_v1.resnet_v1_backbone import (
    ResNet152Backbone,
)
from keras_cv.models.backbones.resnet_v1.resnet_v1_backbone import (
    ResNetBackbone,
)
from keras_cv.utils.train import get_feature_extractor


class ResNetBackboneTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.input_batch = ones(shape=(2, 224, 224, 3))

    def test_valid_call(self):
        model = ResNetBackbone(
            stackwise_filters=[64, 128, 256, 512],
            stackwise_blocks=[2, 2, 2, 2],
            stackwise_strides=[1, 2, 2, 2],
            include_rescaling=False,
        )
        model(self.input_batch)

    def test_valid_call_applications_model(self):
        model = ResNet50Backbone()
        model(self.input_batch)

    def test_valid_call_with_rescaling(self):
        model = ResNetBackbone(
            stackwise_filters=[64, 128, 256, 512],
            stackwise_blocks=[2, 2, 2, 2],
            stackwise_strides=[1, 2, 2, 2],
            include_rescaling=True,
        )
        model(self.input_batch)

    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self):
        model = ResNetBackbone(
            stackwise_filters=[64, 128, 256, 512],
            stackwise_blocks=[2, 2, 2, 2],
            stackwise_strides=[1, 2, 2, 2],
            include_rescaling=False,
        )
        model_output = model(self.input_batch)
        save_path = os.path.join(
            self.get_temp_dir(), "resnet_v1_backbone.keras"
        )
        if use_keras_core():
            model.save(save_path)
        else:
            model.save(save_path, save_format="keras_v3")
        restored_model = load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, ResNetBackbone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)

    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_alias_model(self):
        model = ResNet50Backbone()
        model_output = model(self.input_batch)
        save_path = os.path.join(
            self.get_temp_dir(), "resnet_v1_alias_backbone.keras"
        )
        if use_keras_core():
            model.save(save_path)
        else:
            model.save(save_path, save_format="keras_v3")
        restored_model = load_model(save_path)

        # Check we got the real object back.
        # Note that these aliases serialized as the base class
        self.assertIsInstance(restored_model, ResNetBackbone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)

    def test_create_backbone_model_from_alias_model(self):
        model = ResNet50Backbone(
            include_rescaling=False,
        )
        backbone_model = get_feature_extractor(
            model,
            model.pyramid_level_inputs.values(),
            model.pyramid_level_inputs.keys(),
        )
        inputs = Input(shape=[256, 256, 3])
        outputs = backbone_model(inputs)
        # Resnet50 backbone has 4 level of features (2 ~ 5)
        self.assertLen(outputs, 4)
        self.assertEquals(list(outputs.keys()), [2, 3, 4, 5])
        self.assertEquals(outputs[2].shape, (None, 64, 64, 256))
        self.assertEquals(outputs[3].shape, (None, 32, 32, 512))
        self.assertEquals(outputs[4].shape, (None, 16, 16, 1024))
        self.assertEquals(outputs[5].shape, (None, 8, 8, 2048))

    def test_create_backbone_model_with_level_config(self):
        model = ResNetBackbone(
            stackwise_filters=[64, 128, 256, 512],
            stackwise_blocks=[2, 2, 2, 2],
            stackwise_strides=[1, 2, 2, 2],
            include_rescaling=False,
            input_shape=[256, 256, 3],
        )
        levels = [3, 4]
        layer_names = [model.pyramid_level_inputs[level] for level in [3, 4]]
        backbone_model = get_feature_extractor(model, layer_names, levels)
        inputs = Input(shape=[256, 256, 3])
        outputs = backbone_model(inputs)
        self.assertLen(outputs, 2)
        self.assertEquals(list(outputs.keys()), [3, 4])
        self.assertEquals(outputs[3].shape, (None, 32, 32, 512))
        self.assertEquals(outputs[4].shape, (None, 16, 16, 1024))

    @parameterized.named_parameters(
        ("one_channel", 1),
        ("four_channels", 4),
    )
    def test_application_variable_input_channels(self, num_channels):
        # ResNet50 model
        model = ResNetBackbone(
            stackwise_filters=[64, 128, 256, 512],
            stackwise_blocks=[3, 4, 6, 3],
            stackwise_strides=[1, 2, 2, 2],
            input_shape=(None, None, num_channels),
            include_rescaling=False,
        )
        self.assertEqual(model.output_shape, (None, None, None, 2048))

    @parameterized.named_parameters(
        ("18", ResNet18Backbone),
        ("50", ResNet50Backbone),
        ("101", ResNet101Backbone),
        ("152", ResNet152Backbone),
    )
    def test_specific_arch_forward_pass(self, arch_class):
        backbone = arch_class()
        backbone(tf.random.uniform(shape=[2, 256, 256, 3]))


if __name__ == "__main__":
    tf.test.main()
