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

from tensorflow.keras import layers

from keras_cv.models import utils
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
