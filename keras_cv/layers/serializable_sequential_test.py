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

from copy import deepcopy

import numpy as np

from keras_cv.backend import keras
from keras_cv.layers.serializable_sequential import SerializableSequential
from keras_cv.tests.test_case import TestCase


class TestDetectron2Layers(TestCase):
    def test_sequential_equivalence(self):
        layers = [
            keras.layers.Conv2D(16, 2),
            keras.layers.Flatten(),
            keras.layers.Dense(5),
        ]
        model = SerializableSequential(layers)
        model_keras = keras.Sequential(deepcopy(layers))
        model.build([None, 2, 2, 3])
        model_keras.build([None, 2, 2, 3])
        model.set_weights(model_keras.weights)
        x = np.ones((1, 2, 2, 3))
        x_out = model(x)
        x_expected = model_keras(x)
        self.assertAllClose(x_out, x_expected)
