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

"""
References 1 : https://github.com/sayakpaul/maxim-tf
References 2 : https://github.com/google-research/maxim
"""

import einops
import functools
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.experimental import numpy as tnp

from keras_cv.models import utils

Conv3x3 = functools.partial(layers.Conv2D, kernel_size=(3, 3), padding="same")
Conv1x1 = functools.partial(layers.Conv2D, kernel_size=(1, 1), padding="same")
ConvT_up = functools.partial(
    layers.Conv2DTranspose, kernel_size=(2, 2), strides=(2, 2), padding="same"
)
Conv_down = functools.partial(
    layers.Conv2D, kernel_size=(4, 4), strides=(2, 2), padding="same"
)


def CALayer(
    num_channels: int,
    reduction: int = 4,
    name: str = "channel_attention",
):
    """Squeeze-and-excitation block for channel attention.
    ref: https://arxiv.org/abs/1709.01507
    """

    def apply(x):
        # 2D global average pooling
        y = layers.GlobalAvgPool2D(keepdims=True)(x)
        # Squeeze (in Squeeze-Excitation)
        y = Conv1x1(
            filters=num_channels // reduction, use_bias=True, name=f"{name}_Conv_0"
        )(y)
        y = tf.nn.relu(y)
        # Excitation (in Squeeze-Excitation)
        y = Conv1x1(filters=num_channels, use_bias=True, name=f"{name}_Conv_1")(y)
        y = tf.nn.sigmoid(y)
        return x * y

    return apply


def RCAB(
    num_channels: int,
    reduction: int = 4,
    lrelu_slope: float = 0.2,
    name: str = "residual_ca",
):
    """Residual channel attention block. Contains LN,Conv,lRelu,Conv,SELayer."""

    def apply(x):
        shortcut = x
        x = layers.LayerNormalization(epsilon=1e-06, name=f"{name}_LayerNorm")(x)
        x = Conv3x3(filters=num_channels, use_bias=True, name=f"{name}_conv1")(x)
        x = tf.nn.leaky_relu(x, alpha=lrelu_slope)
        x = Conv3x3(filters=num_channels, use_bias=True, name=f"{name}_conv2")(x)
        x = CALayer(
            num_channels=num_channels,
            reduction=reduction,
            name=f"{name}_channel_attention",
        )(x)
        return x + shortcut

    return apply


def RDCAB(
    num_channels: int,
    reduction: int = 16,
    dropout_rate: float = 0.0,
    name: str = "rdcab",
):
    """Residual dense channel attention block. Used in Bottlenecks."""

    def apply(x):
        y = layers.LayerNormalization(epsilon=1e-06, name=f"{name}_LayerNorm")(x)
        y = MlpBlock(
            mlp_dim=num_channels,
            dropout_rate=dropout_rate,
            name=f"{name}_channel_mixing",
        )(y)
        y = CALayer(
            num_channels=num_channels,
            reduction=reduction,
            name=f"{name}_channel_attention",
        )(y)
        x = x + y
        return x

    return apply


def SAM(
    num_channels: int,
    output_channels: int = 3,
    name: str = "sam",
):

    """Supervised attention module for multi-stage training.
    Introduced by MPRNet [CVPR2021]: https://github.com/swz30/MPRNet
    """

    def apply(x, x_image):
        """Apply the SAM module to the input and num_channels.
        Args:
          x: the output num_channels from UNet decoder with shape (h, w, c)
          x_image: the input image with shape (h, w, 3)
        Returns:
          A tuple of tensors (x1, image) where (x1) is the sam num_channels used for the
            next stage, and (image) is the output restored image at current stage.
        """
        # Get num_channels
        x1 = Conv3x3(filters=num_channels, use_bias=True, name=f"{name}_Conv_0")(x)

        # Output restored image X_s
        if output_channels == 3:
            image = (
                Conv3x3(
                    filters=output_channels, use_bias=True, name=f"{name}_Conv_1"
                )(x)
                + x_image
            )
        else:
            image = Conv3x3(
                filters=output_channels, use_bias=True, name=f"{name}_Conv_1"
            )(x)

        # Get attention maps for num_channels
        x2 = tf.nn.sigmoid(
            Conv3x3(filters=num_channels, use_bias=True, name=f"{name}_Conv_2")(image)
        )

        # Get attended feature maps
        x1 = x1 * x2

        # Residual connection
        x1 = x1 + x
        return x1, image

    return apply


def BlockGatingUnit(name: str = "block_gating_unit"):
    """A SpatialGatingUnit as defined in the gMLP paper.
    The 'spatial' dim is defined as the **second last**.
    If applied on other dims, you should swapaxes first.
    """

    def apply(x):
        u, v = tf.split(x, 2, axis=-1)
        v = layers.LayerNormalization(
            epsilon=1e-06, name=f"{name}_intermediate_layernorm"
        )(v)
        n = K.int_shape(x)[-2]  # get spatial dim
        v = SwapAxes()(v, -1, -2)
        v = layers.Dense(n, use_bias=True, name=f"{name}_Dense_0")(v)
        v = SwapAxes()(v, -1, -2)
        return u * (v + 1.0)

    return apply


def BlockGmlpLayer(
    block_size,
    factor: int = 2,
    dropout_rate: float = 0.0,
    name: str = "block_gmlp",
):
    """Block gMLP layer that performs local mixing of tokens."""

    def apply(x):
        n, h, w, num_channels = (
            K.int_shape(x)[0],
            K.int_shape(x)[1],
            K.int_shape(x)[2],
            K.int_shape(x)[3],
        )
        fh, fw = block_size
        gh, gw = h // fh, w // fw
        x = BlockImages()(x, patch_size=(fh, fw))
        # MLP2: Local (block) mixing part, provides within-block communication.
        y = layers.LayerNormalization(epsilon=1e-06, name=f"{name}_LayerNorm")(x)
        y = layers.Dense(
            num_channels * factor,
            use_bias=True,
            name=f"{name}_in_project",
        )(y)
        y = tf.nn.gelu(y, approximate=True)
        y = BlockGatingUnit(name=f"{name}_BlockGatingUnit")(y)
        y = layers.Dense(
            num_channels,
            use_bias=True,
            name=f"{name}_out_project",
        )(y)
        y = layers.Dropout(dropout_rate)(y)
        x = x + y
        x = UnblockImages()(x, grid_size=(gh, gw), patch_size=(fh, fw))
        return x

    return apply

def BottleneckBlock(
    features: int,
    block_size,
    grid_size,
    num_groups: int = 1,
    block_gmlp_factor: int = 2,
    grid_gmlp_factor: int = 2,
    input_proj_factor: int = 2,
    channels_reduction: int = 4,
    dropout_rate: float = 0.0,
    name: str = "bottleneck_block",
):
    """The bottleneck block consisting of multi-axis gMLP block and RDCAB."""

    def apply(x):
        # input projection
        x = Conv1x1(filters=features, use_bias=True, name=f"{name}_input_proj")(x)
        shortcut_long = x

        for i in range(num_groups):
            x = ResidualSplitHeadMultiAxisGmlpLayer(
                grid_size=grid_size,
                block_size=block_size,
                grid_gmlp_factor=grid_gmlp_factor,
                block_gmlp_factor=block_gmlp_factor,
                input_proj_factor=input_proj_factor,
                dropout_rate=dropout_rate,
                name=f"{name}_SplitHeadMultiAxisGmlpLayer_{i}",
            )(x)
            # Channel-mixing part, which provides within-patch communication.
            x = RDCAB(
                num_channels=features,
                reduction=channels_reduction,
                name=f"{name}_channel_attention_block_1_{i}",
            )(x)

        # long skip-connect
        x = x + shortcut_long
        return x

    return apply

def GridGatingUnit(name: str = "grid_gating_unit"):
    """A SpatialGatingUnit as defined in the gMLP paper.
    The 'spatial' dim is defined as the second last.
    If applied on other dims, you should swapaxes first.
    """

    def apply(x):
        u, v = tf.split(x, 2, axis=-1)
        v = layers.LayerNormalization(
            epsilon=1e-06, name=f"{name}_intermediate_layernorm"
        )(v)
        n = K.int_shape(x)[-3]  # get spatial dim
        v = SwapAxes()(v, -1, -3)
        v = layers.Dense(n, use_bias=True, name=f"{name}_Dense_0")(v)
        v = SwapAxes()(v, -1, -3)
        return u * (v + 1.0)

    return apply


def GridGmlpLayer(
    grid_size,
    factor: int = 2,
    dropout_rate: float = 0.0,
    name: str = "grid_gmlp",
):
    """Grid gMLP layer that performs global mixing of tokens."""

    def apply(x):
        n, h, w, num_channels = (
            K.int_shape(x)[0],
            K.int_shape(x)[1],
            K.int_shape(x)[2],
            K.int_shape(x)[3],
        )
        gh, gw = grid_size
        fh, fw = h // gh, w // gw

        x = BlockImages()(x, patch_size=(fh, fw))
        # gMLP1: Global (grid) mixing part, provides global grid communication.
        y = layers.LayerNormalization(epsilon=1e-06, name=f"{name}_LayerNorm")(x)
        y = layers.Dense(
            num_channels * factor,
            use_bias=True,
            name=f"{name}_in_project",
        )(y)
        y = tf.nn.gelu(y, approximate=True)
        y = GridGatingUnit(name=f"{name}_GridGatingUnit")(y)
        y = layers.Dense(
            num_channels,
            use_bias=True,
            name=f"{name}_out_project",
        )(y)
        y = layers.Dropout(dropout_rate)(y)
        x = x + y
        x = UnblockImages()(x, grid_size=(gh, gw), patch_size=(fh, fw))
        return x

    return apply

def ResidualSplitHeadMultiAxisGmlpLayer(
    block_size,
    grid_size,
    block_gmlp_factor: int = 2,
    grid_gmlp_factor: int = 2,
    input_proj_factor: int = 2,
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
            use_bias=True,
            name=f"{name}_in_project",
        )(x)
        x = tf.nn.gelu(x, approximate=True)

        u, v = tf.split(x, 2, axis=-1)

        # GridGMLPLayer
        u = GridGmlpLayer(
            grid_size=grid_size,
            factor=grid_gmlp_factor,
            dropout_rate=dropout_rate,
            name=f"{name}_GridGmlpLayer",
        )(u)

        # BlockGMLPLayer
        v = BlockGmlpLayer(
            block_size=block_size,
            factor=block_gmlp_factor,
            dropout_rate=dropout_rate,
            name=f"{name}_BlockGmlpLayer",
        )(v)

        x = tf.concat([u, v], axis=-1)

        x = layers.Dense(
            num_channels,
            use_bias=True,
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
            use_bias=True,
            name=f"{name}_in_project",
        )(x)
        x = tf.nn.gelu(x, approximate=True)
        u, v = tf.split(x, 2, axis=-1)

        # Get grid MLP weights
        gh, gw = grid_size
        fh, fw = h // gh, w // gw
        u = BlockImages()(u, patch_size=(fh, fw))
        dim_u = K.int_shape(u)[-3]
        u = SwapAxes()(u, -1, -3)
        u = layers.Dense(dim_u, use_bias=True, name=f"{name}_Dense_0")(u)
        u = SwapAxes()(u, -1, -3)
        u = UnblockImages()(u, grid_size=(gh, gw), patch_size=(fh, fw))

        # Get Block MLP weights
        fh, fw = block_size
        gh, gw = h // fh, w // fw
        v = BlockImages()(v, patch_size=(fh, fw))
        dim_v = K.int_shape(v)[-2]
        v = SwapAxes()(v, -1, -2)
        v = layers.Dense(dim_v, use_bias=True, name=f"{name}_Dense_1")(v)
        v = SwapAxes()(v, -1, -2)
        v = UnblockImages()(v, grid_size=(gh, gw), patch_size=(fh, fw))

        x = tf.concat([u, v], axis=-1)
        x = layers.Dense(num_channels, use_bias=True, name=f"{name}_out_project")(x)
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
            y = ConvT_up(
                filters=features, use_bias=True, name=f"{name}_ConvTranspose_0"
            )(y)

        x = Conv1x1(filters=features, use_bias=True, name=f"{name}_Conv_0")(x)
        n, h, w, num_channels = (
            K.int_shape(x)[0],
            K.int_shape(x)[1],
            K.int_shape(x)[2],
            K.int_shape(x)[3],
        )

        y = Conv1x1(filters=num_channels, use_bias=True, name=f"{name}_Conv_1")(y)

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

def MlpBlock(
    mlp_dim: int,
    dropout_rate: float = 0.0,
    name: str = "mlp_block",
):
    """A 1-hidden-layer MLP block, applied over the last dimension."""

    def apply(x):
        d = K.int_shape(x)[-1]
        x = layers.Dense(mlp_dim, use_bias=True, name=f"{name}_Dense_0")(x)
        x = tf.nn.gelu(x, approximate=True)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(d, use_bias=True, name=f"{name}_Dense_1")(x)
        return x

    return apply


def UpSampleRatio(
    num_channels: int, ratio: float, name: str = "upsample"
):
    """Upsample features given a ratio > 0."""

    def apply(x):
        n, h, w, c = (
            K.int_shape(x)[0],
            K.int_shape(x)[1],
            K.int_shape(x)[2],
            K.int_shape(x)[3],
        )

        # Following `jax.image.resize()`
        x = Resizing(
            height=int(h * ratio),
            width=int(w * ratio),
            method="bilinear",
            antialias=True,
            name=f"{name}_resizing_{K.get_uid('Resizing')}",
        )(x)

        x = Conv1x1(filters=num_channels, use_bias=True, name=f"{name}_Conv_0")(x)
        return x

    return apply


def UNetEncoderBlock(
    num_channels: int,
    block_size,
    grid_size,
    num_groups: int = 1,
    lrelu_slope: float = 0.2,
    block_gmlp_factor: int = 2,
    grid_gmlp_factor: int = 2,
    input_proj_factor: int = 2,
    channels_reduction: int = 4,
    dropout_rate: float = 0.0,
    downsample: bool = True,
    use_global_mlp: bool = True,
    use_cross_gating: bool = False,
    name: str = "unet_encoder",
):
    """Encoder block in MAXIM."""

    def apply(x, skip=None, enc=None, dec=None):
        if skip is not None:
            x = tf.concat([x, skip], axis=-1)

        # convolution-in
        x = Conv1x1(filters=num_channels, use_bias=True, name=f"{name}_Conv_0")(x)
        shortcut_long = x

        for i in range(num_groups):
            if use_global_mlp:
                x = ResidualSplitHeadMultiAxisGmlpLayer(
                    grid_size=grid_size,
                    block_size=block_size,
                    grid_gmlp_factor=grid_gmlp_factor,
                    block_gmlp_factor=block_gmlp_factor,
                    input_proj_factor=input_proj_factor,
                    dropout_rate=dropout_rate,
                    name=f"{name}_SplitHeadMultiAxisGmlpLayer_{i}",
                )(x)
            x = RCAB(
                num_channels=num_channels,
                reduction=channels_reduction,
                lrelu_slope=lrelu_slope,
                name=f"{name}_channel_attention_block_1{i}",
            )(x)

        x = x + shortcut_long

        if enc is not None and dec is not None:
            assert use_cross_gating
            x, _ = CrossGatingBlock(
                features=num_channels,
                block_size=block_size,
                grid_size=grid_size,
                dropout_rate=dropout_rate,
                input_proj_factor=input_proj_factor,
                upsample_y=False,
                name=f"{name}_cross_gating_block",
            )(x, enc + dec)

        if downsample:
            x_down = Conv_down(
                filters=num_channels, use_bias=True, name=f"{name}_Conv_1"
            )(x)
            return x_down, x
        else:
            return x

    return apply


def UNetDecoderBlock(
    num_channels: int,
    block_size,
    grid_size,
    num_groups: int = 1,
    lrelu_slope: float = 0.2,
    block_gmlp_factor: int = 2,
    grid_gmlp_factor: int = 2,
    input_proj_factor: int = 2,
    channels_reduction: int = 4,
    dropout_rate: float = 0.0,
    downsample: bool = True,
    use_global_mlp: bool = True,
    name: str = "unet_decoder",
):

    """Decoder block in MAXIM."""

    def apply(x, bridge=None):
        x = ConvT_up(
            filters=num_channels, use_bias=True, name=f"{name}_ConvTranspose_0"
        )(x)
        x = UNetEncoderBlock(
            num_channels=num_channels,
            num_groups=num_groups,
            lrelu_slope=lrelu_slope,
            block_size=block_size,
            grid_size=grid_size,
            block_gmlp_factor=block_gmlp_factor,
            grid_gmlp_factor=grid_gmlp_factor,
            channels_reduction=channels_reduction,
            use_global_mlp=use_global_mlp,
            dropout_rate=dropout_rate,
            downsample=False,
            name=f"{name}_UNetEncoderBlock_0",
        )(x, skip=bridge)

        return x

    return apply



@tf.keras.utils.register_keras_serializable("maxim")
class BlockImages(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, patch_size):
        bs, h, w, num_channels = (
            K.int_shape(x)[0],
            K.int_shape(x)[1],
            K.int_shape(x)[2],
            K.int_shape(x)[3],
        )

        grid_height, grid_width = h // patch_size[0], w // patch_size[1]

        x = einops.rearrange(
            x,
            "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
            gh=grid_height,
            gw=grid_width,
            fh=patch_size[0],
            fw=patch_size[1],
        )

        return x

    def get_config(self):
        config = super().get_config().copy()
        return config


@tf.keras.utils.register_keras_serializable("maxim")
class UnblockImages(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, grid_size, patch_size):
        x = einops.rearrange(
            x,
            "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
            gh=grid_size[0],
            gw=grid_size[1],
            fh=patch_size[0],
            fw=patch_size[1],
        )

        return x

    def get_config(self):
        config = super().get_config().copy()
        return config


@tf.keras.utils.register_keras_serializable("maxim")
class SwapAxes(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, axis_one, axis_two):
        return tnp.swapaxes(x, axis_one, axis_two)

    def get_config(self):
        config = super().get_config().copy()
        return config


@tf.keras.utils.register_keras_serializable("maxim")
class Resizing(layers.Layer):
    def __init__(self, height, width, antialias=True, method="bilinear", **kwargs):
        super().__init__(**kwargs)
        self.height = height
        self.width = width
        self.antialias = antialias
        self.method = method

    def call(self, x):
        return tf.image.resize(
            x,
            size=(self.height, self.width),
            antialias=self.antialias,
            method=self.method,
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "height": self.height,
                "width": self.width,
                "antialias": self.antialias,
                "method": self.method,
            }
        )
        return config


def MAXIM(
    features: int = 64,
    depth: int = 3,
    num_stages: int = 2,
    num_groups: int = 1,
    num_supervision_scales: int = 1,
    lrelu_slope: float = 0.2,
    use_global_mlp: bool = True,
    use_cross_gating: bool = True,
    high_res_stages: int = 2,
    block_size_hr=(16, 16),
    block_size_lr=(8, 8),
    grid_size_hr=(16, 16),
    grid_size_lr=(8, 8),
    num_bottleneck_blocks: int = 1,
    block_gmlp_factor: int = 2,
    grid_gmlp_factor: int = 2,
    input_proj_factor: int = 2,
    channels_reduction: int = 4,
    num_outputs: int = 3,
    dropout_rate: float = 0.0,
):
    """The MAXIM model function with multi-stage and multi-scale supervision.
    For more model details, please check the CVPR paper:
    MAXIM: MUlti-Axis MLP for Image Processing (https://arxiv.org/abs/2201.02973)
    Attributes:
      features: initial hidden dimension for the input resolution.
      depth: the number of downsampling depth for the model.
      num_stages: how many stages to use. It will also affects the output list.
      num_groups: how many blocks each stage contains.
      use_bias: whether to use bias in all the conv/mlp layers.
      num_supervision_scales: the number of desired supervision scales.
      lrelu_slope: the negative slope parameter in leaky_relu layers.
      use_global_mlp: whether to use the multi-axis gated MLP block (MAB) in each
        layer.
      use_cross_gating: whether to use the cross-gating MLP block (CGB) in the
        skip connections and multi-stage feature fusion layers.
      high_res_stages: how many stages are specificied as high-res stages. The
        rest (depth - high_res_stages) are called low_res_stages.
      block_size_hr: the block_size parameter for high-res stages.
      block_size_lr: the block_size parameter for low-res stages.
      grid_size_hr: the grid_size parameter for high-res stages.
      grid_size_lr: the grid_size parameter for low-res stages.
      num_bottleneck_blocks: how many bottleneck blocks.
      block_gmlp_factor: the input projection factor for block_gMLP layers.
      grid_gmlp_factor: the input projection factor for grid_gMLP layers.
      input_proj_factor: the input projection factor for the MAB block.
      channels_reduction: the channel reduction factor for SE layer.
      num_outputs: the output channels.
      dropout_rate: Dropout rate.
    Returns:
      The output contains a list of arrays consisting of multi-stage multi-scale
      outputs. For example, if num_stages = num_supervision_scales = 3 (the
      model used in the paper), the output specs are: outputs =
      [[output_stage1_scale1, output_stage1_scale2, output_stage1_scale3],
       [output_stage2_scale1, output_stage2_scale2, output_stage2_scale3],
       [output_stage3_scale1, output_stage3_scale2, output_stage3_scale3],]
      The final output can be retrieved by outputs[-1][-1].
    """

    def apply(x):
        n, h, w, c = (
            K.int_shape(x)[0],
            K.int_shape(x)[1],
            K.int_shape(x)[2],
            K.int_shape(x)[3],
        )  # input image shape

        shortcuts = []
        shortcuts.append(x)

        # Get multi-scale input images
        for i in range(1, num_supervision_scales):
            resizing_layer = Resizing(
                height=h // (2 ** i),
                width=w // (2 ** i),
                method="nearest",
                antialias=True,  # Following `jax.image.resize()`.
                name=f"initial_resizing_{K.get_uid('Resizing')}",
            )
            shortcuts.append(resizing_layer(x))

        # store outputs from all stages and all scales
        # Eg, [[(64, 64, 3), (128, 128, 3), (256, 256, 3)],   # Stage-1 outputs
        #      [(64, 64, 3), (128, 128, 3), (256, 256, 3)],]  # Stage-2 outputs
        outputs_all = []
        sam_features, encs_prev, decs_prev = [], [], []

        for idx_stage in range(num_stages):
            # Input convolution, get multi-scale input features
            x_scales = []
            for i in range(num_supervision_scales):
                x_scale = Conv3x3(
                    filters=(2 ** i) * features,
                    use_bias=True,
                    name=f"stage_{idx_stage}_input_conv_{i}",
                )(shortcuts[i])

                # If later stages, fuse input features with SAM features from prev stage
                if idx_stage > 0:
                    # use larger blocksize at high-res stages
                    if use_cross_gating:
                        block_size = (
                            block_size_hr if i < high_res_stages else block_size_lr
                        )
                        grid_size = grid_size_hr if i < high_res_stages else block_size_lr
                        x_scale, _ = CrossGatingBlock(
                            features=(2 ** i) * features,
                            block_size=block_size,
                            grid_size=grid_size,
                            dropout_rate=dropout_rate,
                            input_proj_factor=input_proj_factor,
                            upsample_y=False,
                            name=f"stage_{idx_stage}_input_fuse_sam_{i}",
                        )(x_scale, sam_features.pop())
                    else:
                        x_scale = Conv1x1(
                            filters=(2 ** i) * features,
                            use_bias=True,
                            name=f"stage_{idx_stage}_input_catconv_{i}",
                        )(tf.concat([x_scale, sam_features.pop()], axis=-1))

                x_scales.append(x_scale)

            # start encoder blocks
            encs = []
            x = x_scales[0]  # First full-scale input feature

            for i in range(depth):  # 0, 1, 2
                # use larger blocksize at high-res stages, vice versa.
                block_size = block_size_hr if i < high_res_stages else block_size_lr
                grid_size = grid_size_hr if i < high_res_stages else block_size_lr
                use_cross_gating_layer = True if idx_stage > 0 else False

                # Multi-scale input if multi-scale supervision
                x_scale = x_scales[i] if i < num_supervision_scales else None

                # UNet Encoder block
                enc_prev = encs_prev.pop() if idx_stage > 0 else None
                dec_prev = decs_prev.pop() if idx_stage > 0 else None

                x, bridge = UNetEncoderBlock(
                    num_channels=(2 ** i) * features,
                    num_groups=num_groups,
                    downsample=True,
                    lrelu_slope=lrelu_slope,
                    block_size=block_size,
                    grid_size=grid_size,
                    block_gmlp_factor=block_gmlp_factor,
                    grid_gmlp_factor=grid_gmlp_factor,
                    input_proj_factor=input_proj_factor,
                    channels_reduction=channels_reduction,
                    use_global_mlp=use_global_mlp,
                    dropout_rate=dropout_rate,
                    use_cross_gating=use_cross_gating_layer,
                    name=f"stage_{idx_stage}_encoder_block_{i}",
                )(x, skip=x_scale, enc=enc_prev, dec=dec_prev)

                # Cache skip signals
                encs.append(bridge)

            # Global MLP bottleneck blocks
            for i in range(num_bottleneck_blocks):
                x = BottleneckBlock(
                    block_size=block_size_lr,
                    grid_size=block_size_lr,
                    features=(2 ** (depth - 1)) * features,
                    num_groups=num_groups,
                    block_gmlp_factor=block_gmlp_factor,
                    grid_gmlp_factor=grid_gmlp_factor,
                    input_proj_factor=input_proj_factor,
                    dropout_rate=dropout_rate,
                    channels_reduction=channels_reduction,
                    name=f"stage_{idx_stage}_global_block_{i}",
                )(x)
            # cache global feature for cross-gating
            global_feature = x

            # start cross gating. Use multi-scale feature fusion
            skip_features = []
            for i in reversed(range(depth)):  # 2, 1, 0
                # use larger blocksize at high-res stages
                block_size = block_size_hr if i < high_res_stages else block_size_lr
                grid_size = grid_size_hr if i < high_res_stages else block_size_lr

                # get additional multi-scale signals
                signal = tf.concat(
                    [
                        UpSampleRatio(
                            num_channels=(2 ** i) * features,
                            ratio=2 ** (j - i),
                            name=f"UpSampleRatio_{K.get_uid('UpSampleRatio')}",
                        )(enc)
                        for j, enc in enumerate(encs)
                    ],
                    axis=-1,
                )

                # Use cross-gating to cross modulate features
                if use_cross_gating:
                    skips, global_feature = CrossGatingBlock(
                        features=(2 ** i) * features,
                        block_size=block_size,
                        grid_size=grid_size,
                        input_proj_factor=input_proj_factor,
                        dropout_rate=dropout_rate,
                        upsample_y=True,
                        name=f"stage_{idx_stage}_cross_gating_block_{i}",
                    )(signal, global_feature)
                else:
                    skips = Conv1x1(
                        filters=(2 ** i) * features, use_bias=True, name="Conv_0"
                    )(signal)
                    skips = Conv3x3(
                        filters=(2 ** i) * features, use_bias=True, name="Conv_1"
                    )(skips)

                skip_features.append(skips)

            # start decoder. Multi-scale feature fusion of cross-gated features
            outputs, decs, sam_features = [], [], []
            for i in reversed(range(depth)):
                # use larger blocksize at high-res stages
                block_size = block_size_hr if i < high_res_stages else block_size_lr
                grid_size = grid_size_hr if i < high_res_stages else block_size_lr

                # get multi-scale skip signals from cross-gating block
                signal = tf.concat(
                    [
                        UpSampleRatio(
                            num_channels=(2 ** i) * features,
                            ratio=2 ** (depth - j - 1 - i),
                            name=f"UpSampleRatio_{K.get_uid('UpSampleRatio')}",
                        )(skip)
                        for j, skip in enumerate(skip_features)
                    ],
                    axis=-1,
                )

                # Decoder block
                x = UNetDecoderBlock(
                    num_channels=(2 ** i) * features,
                    num_groups=num_groups,
                    lrelu_slope=lrelu_slope,
                    block_size=block_size,
                    grid_size=grid_size,
                    block_gmlp_factor=block_gmlp_factor,
                    grid_gmlp_factor=grid_gmlp_factor,
                    input_proj_factor=input_proj_factor,
                    channels_reduction=channels_reduction,
                    use_global_mlp=use_global_mlp,
                    dropout_rate=dropout_rate,
                    name=f"stage_{idx_stage}_decoder_block_{i}",
                )(x, bridge=signal)

                # Cache decoder features for later-stage's usage
                decs.append(x)

                # output conv, if not final stage, use supervised-attention-block.
                if i < num_supervision_scales:
                    if idx_stage < num_stages - 1:  # not last stage, apply SAM
                        sam, output = SAM(
                            num_channels=(2 ** i) * features,
                            output_channels=num_outputs,
                            name=f"stage_{idx_stage}_supervised_attention_module_{i}",
                        )(x, shortcuts[i])
                        outputs.append(output)
                        sam_features.append(sam)
                    else:  # Last stage, apply output convolutions
                        output = Conv3x3(
                            num_outputs,
                            use_bias=True,
                            name=f"stage_{idx_stage}_output_conv_{i}",
                        )(x)
                        output = output + shortcuts[i]
                        outputs.append(output)
            # Cache encoder and decoder features for later-stage's usage
            encs_prev = encs[::-1]
            decs_prev = decs

            # Store outputs
            outputs_all.append(outputs)
        return outputs_all

    return apply

MAXIM_CONFIGS = {
    # params: 6.108515000000001 M, GFLOPS: 93.163716608
    "S-1": {
        "features": 32,
        "depth": 3,
        "num_stages": 1,
        "num_groups": 2,
        "num_bottleneck_blocks": 2,
        "block_gmlp_factor": 2,
        "grid_gmlp_factor": 2,
        "input_proj_factor": 2,
        "channels_reduction": 4,
        "name": "s1",
    },
    # params: 13.35383 M, GFLOPS: 206.743273472
    "S-2": {
        "features": 32,
        "depth": 3,
        "num_stages": 2,
        "num_groups": 2,
        "num_bottleneck_blocks": 2,
        "block_gmlp_factor": 2,
        "grid_gmlp_factor": 2,
        "input_proj_factor": 2,
        "channels_reduction": 4,
        "name": "s2",
    },
    # params: 20.599145 M, GFLOPS: 320.32194560000005
    "S-3": {
        "features": 32,
        "depth": 3,
        "num_stages": 3,
        "num_groups": 2,
        "num_bottleneck_blocks": 2,
        "block_gmlp_factor": 2,
        "grid_gmlp_factor": 2,
        "input_proj_factor": 2,
        "channels_reduction": 4,
        "name": "s3",
    },
    # params: 19.361219000000002 M, 308.495712256 GFLOPs
    "M-1": {
        "features": 64,
        "depth": 3,
        "num_stages": 1,
        "num_groups": 2,
        "num_bottleneck_blocks": 2,
        "block_gmlp_factor": 2,
        "grid_gmlp_factor": 2,
        "input_proj_factor": 2,
        "channels_reduction": 4,
        "name": "m1",
    },
    # params: 40.83911 M, 675.25541888 GFLOPs
    "M-2": {
        "features": 64,
        "depth": 3,
        "num_stages": 2,
        "num_groups": 2,
        "num_bottleneck_blocks": 2,
        "block_gmlp_factor": 2,
        "grid_gmlp_factor": 2,
        "input_proj_factor": 2,
        "channels_reduction": 4,
        "name": "m2",
    },
    # params: 62.317001 M, 1042.014666752 GFLOPs
    "M-3": {
        "features": 64,
        "depth": 3,
        "num_stages": 3,
        "num_groups": 2,
        "num_bottleneck_blocks": 2,
        "block_gmlp_factor": 2,
        "grid_gmlp_factor": 2,
        "input_proj_factor": 2,
        "channels_reduction": 4,
        "name": "m3",
    },
}

def Model(variant=None, input_resolution=(256, 256), **kw) -> keras.Model:
    """Factory function to easily create a Model variant like "S".
    Args:
      variant: UNet model variants. Options: 'S-1' | 'S-2' | 'S-3'
          | 'M-1' | 'M-2' | 'M-3'
      input_resolution: Size of the input images.
      **kw: Other UNet config dicts.
    Returns:
      The MAXIM model.
    """

    if variant is not None:
        config = MAXIM_CONFIGS[variant]
        for k, v in config.items():
            kw.setdefault(k, v)

    if "variant" in kw:
        _ = kw.pop("variant")
    if "input_resolution" in kw:
        _ = kw.pop("input_resolution")
    model_name = kw.pop("name")

    maxim_model = MAXIM(**kw)

    inputs = keras.Input((*input_resolution, 3))
    outputs = maxim_model(inputs)
    final_model = keras.Model(inputs, outputs, name=f"{model_name}_model")

    return final_model

model = Model(variant="S-1")
print(model.summary())