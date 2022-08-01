# Copyright 2022 The KerasCV Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility functions for models"""

from tensorflow import keras
from tensorflow.keras import layers


def parse_model_inputs(input_shape, input_tensor):
    if input_tensor is None:
        return layers.Input(shape=input_shape)
    else:
        if not keras.backend.is_keras_tensor(input_tensor):
            return layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            return input_tensor
