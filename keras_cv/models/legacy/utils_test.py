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
"""Tests for KerasCV model utils."""

from keras_cv.backend import keras
from keras_cv.backend.keras import layers
from keras_cv.models.legacy import utils
from keras_cv.tests.test_case import TestCase


class ModelUtilTestCase(TestCase):
    def test_parse_model_inputs(self):
        input_shape = (224, 244, 3)

        inputs = utils.parse_model_inputs(input_shape, None)
        self.assertEqual(inputs.shape.as_list(), list((None,) + input_shape))

        input_tensor = layers.Input(shape=input_shape)
        self.assertIs(
            utils.parse_model_inputs(input_shape, input_tensor), input_tensor
        )

    def test_as_backbone_missing_backbone_level_outputs(self):
        model = keras.models.Sequential()
        model.add(layers.Conv2D(64, kernel_size=3, input_shape=(16, 16, 3)))
        model.add(
            layers.Conv2D(
                32,
                kernel_size=3,
            )
        )
        model.add(layers.Dense(10))
        with self.assertRaises(ValueError):
            utils.as_backbone(model)

    def test_as_backbone_util(self):
        inp = layers.Input((16, 16, 3))
        _backbone_level_outputs = {}

        x = layers.Conv2D(64, kernel_size=3, input_shape=(16, 16, 3))(inp)
        _backbone_level_outputs[2] = x

        x = layers.Conv2D(
            32,
            kernel_size=3,
        )(x)
        _backbone_level_outputs[3] = x

        out = layers.Dense(10)(x)
        _backbone_level_outputs[4] = out

        model = keras.models.Model(inputs=inp, outputs=out)

        # when model has _backbone_level_outputs, it should not raise an error
        model._backbone_level_outputs = _backbone_level_outputs

        backbone = utils.as_backbone(model)
        self.assertEqual(len(backbone.outputs), 3)
