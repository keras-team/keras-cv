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

from keras_cv import use_keras_core

if use_keras_core():
    from keras_core import backend
    from keras_core import layers
    from keras_core.backend import is_keras_tensor
else:
    from keras import backend
    from keras import layers
    from keras.backend import is_keras_tensor


def get_tensor_input_name(tensor):
    if use_keras_core():
        return tensor._keras_history.operation.name
    else:
        return tensor.node.layer.name


def parse_model_inputs(input_shape, input_tensor):
    if input_tensor is None:
        return layers.Input(shape=input_shape)
    else:
        if not is_keras_tensor(input_tensor):
            return layers.Input(tensor=input_tensor, shape=input_shape)
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
    input_size = backend.int_shape(inputs)[img_dim : (img_dim + 2)]
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
