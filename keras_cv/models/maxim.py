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

import einops
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.experimental import numpy as tnp

from keras_cv.models import utils


def SELayer(filters, name = "SELayer"):
    """SE layer from  Squeeze-and-excitation networks."""

    def apply(x):
        y = layers.GlobalAveragePooling2D(keepdims=True)(x)
        y = layers.Conv2D(filters=filters // 4, kernel_size=(1,1) ,use_bias=True, padding="same", name=f"{name}_Conv_0")(y)
        y = tf.nn.relu(y)
        y = layers.Conv2D(filters=filters, kernel_size=(1,1), use_bias=True, padding="same", name=f"{name}_Conv_1")(y)
        y = tf.nn.sigmoid(y)
        return x * y

    return apply


def RCAB(filters, name = "RCAB_block"):
    """from the paper : (LayerNormConv-LeakyReLU-Conv-SE)"""

    def apply(x):
        shortcut = x
        x = layers.LayerNormalization(epsilon=1e-06, name=f"{name}_LayerNorm")(x)
        x = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", use_bias=True, name=f"{name}_conv1")(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", use_bias=True, name=f"{name}_conv2")(x)
        x = SELayer(filters=filters,reduction=4,use_bias=True,name=f"{name}_channel_attention",)(x)
        return x + shortcut

    return apply


def RDCAB(filters, dropout_rate = 0.0, name = "RDCAB_layer"):

    def apply(x):
        y = layers.LayerNormalization(epsilon=1e-06, name=f"{name}_LayerNorm")(x)
        dim = K.int_shape(y)[-1]
        y = layers.Dense(filters, use_bias=True, name=f"{name}_Dense_0")(y)
        y = tf.nn.gelu(x, approximate=True)
        y = layers.Dropout(dropout_rate)(x)
        y = layers.Dense(dim, use_bias=True, name=f"{name}_Dense_1")(x)
        y = SELayer(filters=filters, reduction=16, use_bias=True, name=f"{name}_channel_attention")(y)
        x = x + y
        return x

    return apply


def SAM(filters, output_channels=3, name = "SAM_block"):
    def apply(x, x_image):
        x1 = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", use_bias=True, name=f"{name}_Conv_0")(x)
        if output_channels == 3:
            image = (layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", use_bias=True, name=f"{name}_Conv_1")(x)
                + x_image
            )
        else:
            image = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", use_bias=True, name=f"{name}_Conv_0")(x)
        x2 = tf.nn.sigmoid(layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", use_bias=True, name=f"{name}_Conv_0")(image))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, image

    return apply


def BlockGatingUnit(name = "block_gating_unit"):

    def apply(x):
        u, v = tf.split(x, 2, axis=-1)
        v = layers.LayerNormalization(
            epsilon=1e-06, name=f"{name}_intermediate_layernorm"
        )(v)
        n = K.int_shape(x)[-2]
        v = tnp.swapaxes(v, -1, -2)
        v = layers.Dense(n, use_bias=True, name=f"{name}_Dense_0")(v)
        v = tnp.swapaxes(v, -1, -2)
        return u * (v + 1.0)

    return apply


def BlockGmlpLayer(block_size, dropout_rate = 0.0, name = "block_gmlp"):

    def apply(x):
        n, h, w, num_channels = (
            K.int_shape(x)[0],
            K.int_shape(x)[1],
            K.int_shape(x)[2],
            K.int_shape(x)[3],
        )
        fh, fw = block_size
        gh, gw = h // fh, w // fw

        grid_height, grid_width = h // fh, w // fw
        x = einops.rearrange(x,"n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c", gh=grid_height, gw=grid_width, fh=fh, fw=fw,)
        y = layers.LayerNormalization(epsilon=1e-06, name=f"{name}_LayerNorm")(x)
        y = layers.Dense(num_channels * 2, use_bias=True, name=f"{name}_in_project")(y)
        y = tf.nn.gelu(y, approximate=True)
        y = BlockGatingUnit(use_bias=True, name=f"{name}_BlockGatingUnit")(y)
        y = layers.Dense(num_channels, use_bias=True, name=f"{name}_out_project",)(y)
        y = layers.Dropout(dropout_rate)(y)
        x = x + y

        x = einops.rearrange(x,"n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",gh=gh,gw=gw,fh=fh,fw=fw)
        return x

    return apply