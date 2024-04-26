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

from keras_cv.src.backend import keras
from keras_cv.src.backend.config import keras_3


def get_tensor_input_name(tensor):
    if keras_3():
        return tensor._keras_history.operation.name
    else:
        return tensor.node.layer.name


def parse_model_inputs(input_shape, input_tensor, **kwargs):
    if input_tensor is None:
        return keras.layers.Input(shape=input_shape, **kwargs)
    else:
        if not keras.backend.is_keras_tensor(input_tensor):
            return keras.layers.Input(
                tensor=input_tensor, shape=input_shape, **kwargs
            )
        else:
            return input_tensor


def correct_pad_downsample(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    Args:
        inputs: Input tensor.
        kernel_size: An integer or tuple/list of 2 integers.

    Returns:
        A tuple.
    """
    img_dim = 1
    input_size = inputs.shape[img_dim : (img_dim + 2)]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return (
        (correct[0] - adjust[0], correct[0]),
        (correct[1] - adjust[1], correct[1]),
    )
