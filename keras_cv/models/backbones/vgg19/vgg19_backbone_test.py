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

from keras_cv.models.backbones.vgg19.vgg19_backbone import VGG19Backbone
from tensorflow import keras

class VGG19BackboneTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.input_batch = tf.ones(shape=(2, 224, 224, 3))

    def test_valid_call(self):
        model = VGG19Backbone(
            include_rescaling=False
        )
        model(self.input_batch)

    def test_valid_call_applications_model(self):
        model = VGG19Backbone()
        model(self.input_batch)
    
    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras")
    )
    def test_saved_alias_model(self, save_format, filename):
        model = VGG19Backbone()

        model_output = model(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(), filename)
        model.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        self.assertIsInstance(restored_model, VGG19Backbone)

        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)
        
if __name__ == "__main__":
    tf.test.main()
