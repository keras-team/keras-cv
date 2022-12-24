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

def block_images_einops(x, patch_size):
  """Image to patches."""
  batch, height, width, channels = x.shape
  grid_height = height // patch_size[0]
  grid_width = width // patch_size[1]
  x = einops.rearrange(
      x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
      gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
  return x

def unblock_images_einops(x, grid_size, patch_size):
  """patches to images."""
  x = einops.rearrange(
      x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
      gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
  return x


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

        x = block_images_einops(x, patch_size=(fh, fw))
        y = layers.LayerNormalization(epsilon=1e-06, name=f"{name}_LayerNorm")(x)
        y = layers.Dense(num_channels * 2, use_bias=True, name=f"{name}_in_project")(y)
        y = tf.nn.gelu(y, approximate=True)
        y = BlockGatingUnit(use_bias=True, name=f"{name}_BlockGatingUnit")(y)
        y = layers.Dense(num_channels, use_bias=True, name=f"{name}_out_project",)(y)
        y = layers.Dropout(dropout_rate)(y)
        x = x + y

        x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
        return x

    return apply


def GridGatingUnit(name: str = "grid_gating_unit"):

    def apply(x):
        u, v = tf.split(x, 2, axis=-1)
        v = layers.LayerNormalization(
            epsilon=1e-06, name=f"{name}_intermediate_layernorm"
        )(v)
        n = K.int_shape(x)[-3]  # get spatial dim
        v = tnp.swapaxes(v, -1, -3)
        v = layers.Dense(n, use_bias=True, name=f"{name}_Dense_0")(v)
        v = tnp.swapaxes(v, -1, -3)
        return u * (v + 1.0)

    return apply


def GridGmlpLayer(grid_size, dropout_rate: float = 0.0, name: str = "grid_gmlp"):
    def apply(x):
        n, h, w, num_channels = (
            K.int_shape(x)[0],
            K.int_shape(x)[1],
            K.int_shape(x)[2],
            K.int_shape(x)[3],
        )
        gh, gw = grid_size
        fh, fw = h // gh, w // gw

        x = block_images_einops(x, patch_size=(fh, fw))
        y = layers.LayerNormalization(epsilon=1e-06, name=f"{name}_LayerNorm")(x)
        y = layers.Dense(num_channels * 2,
            use_bias=True,
            name=f"{name}_in_project",
        )(y)
        y = tf.nn.gelu(y, approximate=True)
        y = GridGatingUnit(use_bias=True, name=f"{name}_GridGatingUnit")(y)
        y = layers.Dense(
            num_channels,
            use_bias=True,
            name=f"{name}_out_project",
        )(y)
        y = layers.Dropout(dropout_rate)(y)
        x = x + y

        x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
        return x

    return apply


def ResidualSplitHeadMultiAxisGmlpLayer(
    block_size,
    grid_size,
    block_gmlp_factor: int = 2,
    grid_gmlp_factor: int = 2,
    input_proj_factor: int = 2,
    use_bias: bool = True,
    dropout_rate: float = 0.0,
    name: str = "residual_split_head_maxim",
):
    """The multi-axis gated MLP block."""

    def apply(x):
        shortcut = x
        n, h, w, num_channels = (
            K.int_shape(x)[0],
            K.int_shape(x)[1],
            K.int_shape(x)[2],
            K.int_shape(x)[3],
        )
        x = layers.LayerNormalization(epsilon=1e-06, name=f"{name}_LayerNorm_in")(x)

        x = layers.Dense(
            int(num_channels) * input_proj_factor,
            use_bias=use_bias,
            name=f"{name}_in_project",
        )(x)
        x = tf.nn.gelu(x, approximate=True)

        u, v = tf.split(x, 2, axis=-1)

        # GridGMLPLayer
        u = GridGmlpLayer(
            grid_size=grid_size,
            factor=grid_gmlp_factor,
            use_bias=use_bias,
            dropout_rate=dropout_rate,
            name=f"{name}_GridGmlpLayer",
        )(u)

        # BlockGMLPLayer
        v = BlockGmlpLayer(
            block_size=block_size,
            factor=block_gmlp_factor,
            use_bias=use_bias,
            dropout_rate=dropout_rate,
            name=f"{name}_BlockGmlpLayer",
        )(v)

        x = tf.concat([u, v], axis=-1)

        x = layers.Dense(
            num_channels,
            use_bias=use_bias,
            name=f"{name}_out_project",
        )(x)
        x = layers.Dropout(dropout_rate)(x)
        x = x + shortcut
        return x

    return apply


def GetSpatialGatingWeights(
    features: int,
    block_size,
    grid_size,
    input_proj_factor: int = 2,
    dropout_rate: float = 0.0,
    use_bias: bool = True,
    name: str = "spatial_gating",
):

    """Get gating weights for cross-gating MLP block."""

    def apply(x):
        n, h, w, num_channels = (
            K.int_shape(x)[0],
            K.int_shape(x)[1],
            K.int_shape(x)[2],
            K.int_shape(x)[3],
        )

        # input projection
        x = layers.LayerNormalization(epsilon=1e-06, name=f"{name}_LayerNorm_in")(x)
        x = layers.Dense(
            num_channels * input_proj_factor,
            use_bias=use_bias,
            name=f"{name}_in_project",
        )(x)
        x = tf.nn.gelu(x, approximate=True)
        u, v = tf.split(x, 2, axis=-1)

        # Get grid MLP weights
        gh, gw = grid_size
        fh, fw = h // gh, w // gw

        u = block_images_einops(u, patch_size=(fh, fw))
        dim_u = K.int_shape(u)[-3]
        u = tnp.swapaxes(u, -1, -3)
        u = layers.Dense(dim_u, use_bias=use_bias, name=f"{name}_Dense_0")(u)
        u = tnp.swapaxes(u, -1, -3)
        u = unblock_images_einops(u, grid_size=(gh, gw), patch_size=(fh, fw))

        # Get Block MLP weights
        fh, fw = block_size
        gh, gw = h // fh, w // fw
        v = block_images_einops(v, patch_size=(fh, fw))
        dim_v = K.int_shape(v)[-2]
        u = tnp.swapaxes(u, -1, -2)
        v = layers.Dense(dim_v, use_bias=use_bias, name=f"{name}_Dense_1")(v)
        u = tnp.swapaxes(v, -1, -2)
        v = unblock_images_einops(v, grid_size=(gh, gw), patch_size=(fh, fw))

        x = tf.concat([u, v], axis=-1)
        x = layers.Dense(num_channels, use_bias=use_bias, name=f"{name}_out_project")(x)
        x = layers.Dropout(dropout_rate)(x)
        return x

    return apply


def CrossGatingBlock(
    features: int,
    block_size,
    grid_size,
    dropout_rate: float = 0.0,
    input_proj_factor: int = 2,
    upsample_y: bool = True,
    name: str = "cross_gating",
):

    """Cross-gating MLP block."""

    def apply(x, y):
        # Upscale Y signal, y is the gating signal.
        if upsample_y:
            y = layers.Conv2DTranspose(
                filters=features, kernel_size=(2, 2), strides=(2, 2), padding="same", use_bias=True, name=f"{name}_ConvTranspose_0"
            )(y)

        x = layers.Conv2D(filters=features, kernel_size=(1, 1), padding="same", use_bias=True, name=f"{name}_Conv_0")(x)
        n, h, w, num_channels = (
            K.int_shape(x)[0],
            K.int_shape(x)[1],
            K.int_shape(x)[2],
            K.int_shape(x)[3],
        )

        y = layers.Conv2D(filters=num_channels, kernel_size=(1, 1), padding="same", use_bias=True, name=f"{name}_Conv_1")(y)

        shortcut_x = x
        shortcut_y = y

        # Get gating weights from X
        x = layers.LayerNormalization(epsilon=1e-06, name=f"{name}_LayerNorm_x")(x)
        x = layers.Dense(num_channels, use_bias=True, name=f"{name}_in_project_x")(x)
        x = tf.nn.gelu(x, approximate=True)
        gx = GetSpatialGatingWeights(
            features=num_channels,
            block_size=block_size,
            grid_size=grid_size,
            dropout_rate=dropout_rate,
            use_bias=True,
            name=f"{name}_SplitHeadMultiAxisGating_x",
        )(x)

        # Get gating weights from Y
        y = layers.LayerNormalization(epsilon=1e-06, name=f"{name}_LayerNorm_y")(y)
        y = layers.Dense(num_channels, use_bias=True, name=f"{name}_in_project_y")(y)
        y = tf.nn.gelu(y, approximate=True)
        gy = GetSpatialGatingWeights(
            features=num_channels,
            block_size=block_size,
            grid_size=grid_size,
            dropout_rate=dropout_rate,
            use_bias=True,
            name=f"{name}_SplitHeadMultiAxisGating_y",
        )(y)

        # Apply cross gating: X = X * GY, Y = Y * GX
        y = y * gx
        y = layers.Dense(num_channels, use_bias=True, name=f"{name}_out_project_y")(y)
        y = layers.Dropout(dropout_rate)(y)
        y = y + shortcut_y

        x = x * gy  # gating x using y
        x = layers.Dense(num_channels, use_bias=True, name=f"{name}_out_project_x")(x)
        x = layers.Dropout(dropout_rate)(x)
        x = x + y + shortcut_x  # get all aggregated signals
        return x, y

    return apply