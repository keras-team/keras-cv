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


"""ConvMixer models for Keras.

References:
- [Patches Are All You Need?](https://arxiv.org/abs/2201.09792)
"""

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers

from keras_cv.models import utils

MODEL_CONFIGS = {
    "ConvMixer_1536_20": {
        "dim": 1536,
        "depth": 20,
        "patch_size": 7,
        "kernel_size": 9
    },
    "ConvMixer_1536_24": {
        "dim": 1536,
        "depth": 24,
        "patch_size": 14,
        "kernel_size": 9,
    },
    "ConvMixer_768_32": {
        "dim": 768,
        "depth": 32,
        "patch_size": 7,
        "kernel_size": 7,
    },
    "ConvMixer_1024_16": {
        "dim": 1024,
        "depth": 16,
        "patch_size": 7,
        "kernel_size": 9,
    },
    "ConvMixer_512_16": {
        "dim": 512,
        "depth": 16,
        "patch_size": 7,
        "kernel_size": 8,
    },
}

def CovnMixer_Layer(inputs, dim, kernel_size):
    """CovnMixer Layer.
    Args:
        inputs: Input tensor.
        dim: integer, filters of the layer in a block.
        kernel_size: integer, kernel size of the Conv2d layers.
    Returns:
        Output tensor for the CovnMixer Layer.
    """
    residual = inputs
    x = tf.keras.layers.DepthwiseConv2D(kernel_size = kernel_size, padding="same")(
        inputs
    )
    x = tf.nn.gelu(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, residual])

    x = tf.keras.layers.Conv2D(dim, kernel_size=1)(x)
    x = tf.nn.gelu(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def patch_embed(inputs, dim, patch_size):
    """Extract Patch Embedding.
    Args:
        inputs: Input tensor.
        patch_size: integer, Size of patches.
    Returns:
        Output tensor for the patch embed.
    """
    x = tf.keras.layers.Conv2D(filters=dim, kernel_size=patch_size, strides=patch_size)(
        inputs
    )
    x = tf.nn.gelu(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x