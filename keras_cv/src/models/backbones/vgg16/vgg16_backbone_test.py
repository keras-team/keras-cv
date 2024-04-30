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

import numpy as np
import pytest

from keras_cv.src.backend import keras
from keras_cv.src.models import VGG16Backbone
from keras_cv.src.tests.test_case import TestCase


class VGG16BackboneTest(TestCase):
    def setUp(self):
        self.img_input = np.ones((2, 224, 224, 3), dtype="float32")

    def test_valid_call(self):
        model = VGG16Backbone(
            input_shape=(224, 224, 3),
            include_top=False,
            include_rescaling=False,
            pooling="avg",
        )
        model(self.img_input)

    def test_valid_call_with_rescaling(self):
        model = VGG16Backbone(
            input_shape=(224, 224, 3),
            include_top=False,
            include_rescaling=True,
            pooling="avg",
        )
        model(self.img_input)

    def test_valid_call_with_top(self):
        model = VGG16Backbone(
            input_shape=(224, 224, 3),
            include_top=True,
            include_rescaling=False,
            num_classes=2,
        )
        model(self.img_input)

    @pytest.mark.large
    def test_saved_model(self):
        model = VGG16Backbone(
            input_shape=(224, 224, 3),
            include_top=False,
            include_rescaling=False,
            num_classes=2,
            pooling="avg",
        )
        model_output = model(self.img_input)
        save_path = os.path.join(self.get_temp_dir(), "vgg16.keras")
        model.save(save_path)
        restored_model = keras.models.load_model(save_path)

        # Check the restored model is instance of VGG16Backbone
        self.assertIsInstance(restored_model, VGG16Backbone)

        # Check if the restored model gives the same output
        restored_model_output = restored_model(self.img_input)
        self.assertAllClose(model_output, restored_model_output)
