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

from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX002Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX004Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX006Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX008Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX016Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX032Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX040Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX064Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX080Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX120Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX160Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetX320Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY002Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY004Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY006Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY008Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY016Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY032Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY040Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY064Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY080Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY120Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY160Backbone
from keras_cv.models.backbones.regnet.regnet_aliases import RegNetY320Backbone
from keras_cv.models.backbones.regnet.regnet_backbone import RegNetBackBone


class RegNetBackboneTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.input_batch = tf.ones(shape=(2, 224, 224, 3))

    def test_valid_call(self):
        model = RegNetBackBone(
            depths=[1, 1, 4, 7],
            widths=[24, 56, 152, 368],
            group_width=8,
            default_size=224,
            include_rescaling=False,
        )
        model(self.input_batch)

    def test_valid_call_applications_model(self):
        model = RegNetX002Backbone()
        model(self.input_batch)

    def test_valid_call_with_rescaling(self):
        model = RegNetBackBone(
            depths=[1, 1, 4, 7],
            widths=[24, 56, 152, 368],
            group_width=8,
            default_size=224,
            include_rescaling=True,
        )
        model(self.input_batch)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        model = RegNetBackBone(
            depths=[1, 1, 4, 7],
            widths=[24, 56, 152, 368],
            group_width=8,
            default_size=224,
            include_rescaling=False,
        )
        model_output = model(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(), filename)
        model.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, RegNetBackBone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_alias_model(self, save_format, filename):
        model = RegNetX002Backbone()
        model_output = model(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(), filename)
        model.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        # Note that these aliases serialized as the base class
        self.assertIsInstance(restored_model, RegNetBackBone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)

    def test_model_backbone_layer_names_stability(self):
        model = RegNetBackBone(
            depths=[1, 1, 4, 7],
            widths=[24, 56, 152, 368],
            group_width=8,
            default_size=224,
            include_rescaling=False,
        )
        model_2 = RegNetBackBone(
            depths=[1, 1, 4, 7],
            widths=[24, 56, 152, 368],
            group_width=8,
            default_size=224,
            include_rescaling=False,
        )
        layers_1 = model.layers
        layers_2 = model_2.layers
        for i in range(len(layers_1)):
            if "input" in layers_1[i].name:
                continue
            self.assertEquals(layers_1[i].name, layers_2[i].name)

    @parameterized.named_parameters(
        ("one_channel", 1),
        ("four_channels", 4),
    )
    def test_application_variable_input_channels(self, num_channels):
        # RegNetX002 model
        model = RegNetBackBone(
            depths=[1, 1, 4, 7],
            widths=[24, 56, 152, 368],
            group_width=8,
            default_size=224,
            input_shape=(None, None, num_channels),
            include_rescaling=False,
        )
        self.assertEqual(model.output_shape, (None, None, None, 2048))

    @parameterized.named_parameters(
        ("x002", RegNetX002Backbone),
        ("x004", RegNetX004Backbone),
        ("x006", RegNetX006Backbone),
        ("x008", RegNetX008Backbone),
        ("x016", RegNetX016Backbone),
        ("x032", RegNetX032Backbone),
        ("x040", RegNetX040Backbone),
        ("x064", RegNetX064Backbone),
        ("x080", RegNetX080Backbone),
        ("x120", RegNetX120Backbone),
        ("x160", RegNetX160Backbone),
        ("x320", RegNetX320Backbone),
        ("y002", RegNetY002Backbone),
        ("y004", RegNetY004Backbone),
        ("y006", RegNetY006Backbone),
        ("y008", RegNetY008Backbone),
        ("y016", RegNetY016Backbone),
        ("y032", RegNetY032Backbone),
        ("y040", RegNetY040Backbone),
        ("y064", RegNetY064Backbone),
        ("y080", RegNetY080Backbone),
        ("y120", RegNetY120Backbone),
        ("y160", RegNetY160Backbone),
        ("y320", RegNetY320Backbone),
    )
    def test_specific_arch_forward_pass(self, arch_class):
        backbone = arch_class()
        backbone(tf.random.uniform(shape=[2, 256, 256, 3]))


if __name__ == "__main__":
    tf.test.main()
