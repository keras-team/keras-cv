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

from keras_cv.models import resnet_v2

from .models_test import ModelsTest

MODEL_LIST = [
    (resnet_v2.ResNet18V2, 512, {}),
]

"""
Below are other configurations that we omit from our CI but that can/should
be tested manually when making changes to this model.
(resnet_v2.ResNet34V2, 512, {}),
(resnet_v2.ResNet50V2, 2048, {}),
(resnet_v2.ResNet101V2, 2048, {}),
(resnet_v2.ResNet152V2, 2048, {}),
"""


class ResNetV2Test(ModelsTest, tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(*MODEL_LIST)
    def test_application_base(self, app, _, args):
        super()._test_application_base(app, _, args)

    @parameterized.parameters(*MODEL_LIST)
    def test_application_with_rescaling(self, app, last_dim, args):
        super()._test_application_with_rescaling(app, last_dim, args)

    @parameterized.parameters(*MODEL_LIST)
    def test_application_pooling(self, app, last_dim, args):
        super()._test_application_pooling(app, last_dim, args)

    @parameterized.parameters(*MODEL_LIST)
    def test_application_variable_input_channels(self, app, last_dim, args):
        super()._test_application_variable_input_channels(app, last_dim, args)

    @parameterized.parameters(*MODEL_LIST)
    def test_model_can_be_used_as_backbone(self, app, last_dim, args):
        super()._test_model_can_be_used_as_backbone(app, last_dim, args)

    def test_model_backbone_layer_names_stability(self):
        model = resnet_v2.ResNet50V2(
            include_rescaling=False,
            include_top=False,
            classes=2048,
            input_shape=[256, 256, 3],
        )
        model_2 = resnet_v2.ResNet50V2(
            include_rescaling=False,
            include_top=False,
            classes=2048,
            input_shape=[256, 256, 3],
        )
        layers_1 = model.layers
        layers_2 = model_2.layers
        for i in range(len(layers_1)):
            if "input" in layers_1[i].name:
                continue
            self.assertEquals(layers_1[i].name, layers_2[i].name)

    def test_create_backbone_model_from_application_model(self):
        model = resnet_v2.ResNet50V2(
            include_rescaling=False,
            include_top=False,
            classes=2048,
            input_shape=[256, 256, 3],
        )
        backbone_model = model.as_backbone()
        inputs = tf.keras.Input(shape=[256, 256, 3])
        outputs = backbone_model(inputs)
        # Resnet50 backbone has 4 level of features (2 ~ 5)
        self.assertLen(outputs, 4)
        self.assertEquals(list(outputs.keys()), [2, 3, 4, 5])
        self.assertEquals(outputs[2].shape, [None, 64, 64, 256])
        self.assertEquals(outputs[3].shape, [None, 32, 32, 512])
        self.assertEquals(outputs[4].shape, [None, 16, 16, 1024])
        self.assertEquals(outputs[5].shape, [None, 8, 8, 2048])

    def test_create_backbone_model_with_level_config(self):
        model = resnet_v2.ResNet50V2(
            include_rescaling=False,
            include_top=False,
            classes=2048,
            input_shape=[256, 256, 3],
        )
        backbone_model = model.as_backbone(min_level=3, max_level=4)
        inputs = tf.keras.Input(shape=[256, 256, 3])
        outputs = backbone_model(inputs)
        self.assertLen(outputs, 2)
        self.assertEquals(list(outputs.keys()), [3, 4])
        self.assertEquals(outputs[3].shape, [None, 32, 32, 512])
        self.assertEquals(outputs[4].shape, [None, 16, 16, 1024])


if __name__ == "__main__":
    tf.test.main()
