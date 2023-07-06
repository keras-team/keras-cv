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
from keras_cv.models.backbones.vit.vit_backbone import ViTBackbone
from keras_cv.models.backbones.vit.vit_aliases import ViTTiny16Backbone
from keras_cv.models.backbones.vit.vit_aliases import ViTS16Backbone
from keras_cv.models.backbones.vit.vit_aliases import ViTB16Backbone
from keras_cv.models.backbones.vit.vit_aliases import ViTL16Backbone
from keras_cv.models.backbones.vit.vit_aliases import ViTH16Backbone
from keras_cv.models.backbones.vit.vit_aliases import ViTTiny32Backbone
from keras_cv.models.backbones.vit.vit_aliases import ViTS32Backbone
from keras_cv.models.backbones.vit.vit_aliases import ViTB32Backbone
from keras_cv.models.backbones.vit.vit_aliases import ViTL32Backbone
from keras_cv.models.backbones.vit.vit_aliases import ViTH16Backbone

from keras_cv.utils.train import get_feature_extractor

from keras_cv.models.models_test import ModelsTest

MODEL_LIST = [
    (vit.ViTS16, 384, {"input_shape": (224, 224, 3)}),
]

"""
Below are other configurations that we omit from our CI but that can/should
be tested manually when making changes to this model.
(vit.ViTTiny16, 192, {"input_shape": (224, 224, 3)}),
(vit.ViTB16, 768, {"input_shape": (224, 224, 3)}),
(vit.ViTL16, 1024, {"input_shape": (224, 224, 3)}),
(vit.ViTH16, 1280, {"input_shape": (224, 224, 3)}),
(vit.ViTTiny32, 192, {"input_shape": (224, 224, 3)}),
(vit.ViTS32, 384, {"input_shape": (224, 224, 3)}),
(vit.ViTB32, 768, {"input_shape": (224, 224, 3)}),
(vit.ViTL32, 1024, {"input_shape": (224, 224, 3)}),
(vit.ViTH32, 1280, {"input_shape": (224, 224, 3)}),
"""
class ViTBackboneTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.input_batch = tf.ones(shape=(2, 224, 224, 3))
    
    def test_valid_call(self):
        model = ViTBackbone(
            include_rescaling=False
        )
        model(self.input_batch)

    def test_valid_call_applications_model(self):
        model = ViTTiny16Backbone()
        model(self.input_batch)

    def test_valid_call_with_rescaling(self):
        model = ViTBackbone(
            include_rescaling=True
        )
        model(self.input_batch)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        model = ViTBackbone(
            include_rescaling=False
        )
        model_output = model(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(), filename)
        model.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        self.assertIsInstance(restored_model, ViTBackbone)

        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_alias_model(self, save_format, filename):
        model = ViTTiny16Backbone()
        model_output = model(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(), filename)
        model.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        self.assertIsInstance(restored_model, ViTBackbone)

        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)

    def test_model_backbone_layer_names_stability(self):
        model_1 = ViTBackbone(

        )
        model_2 = ViTBackbone(

        )
        layers_1 = model_1.layers
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
        model = ViTBackbone(
            input_shape=(None, None, num_channels),
            include_rescaling=False,
        )
        self.assertEqual(model.output_shape, (None, None, None, 2048))
        model(self.input_batch)

if __name__ == "__main__":
    tf.test.main()
