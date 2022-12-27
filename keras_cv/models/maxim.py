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
        x = layers.Conv2D(filters=features, kernel_size=(1, 1), padding="same", use_bias=True, name=f"{name}_input_proj")(x)
        shortcut_long = x

        for i in range(num_groups):
            x = ResidualSplitHeadMultiAxisGmlpLayer(
                grid_size=grid_size,
                block_size=block_size,
                grid_gmlp_factor=grid_gmlp_factor,
                block_gmlp_factor=block_gmlp_factor,
                input_proj_factor=input_proj_factor,
                use_bias=True,
                dropout_rate=dropout_rate,
                name=f"{name}_SplitHeadMultiAxisGmlpLayer_{i}",
            )(x)
            # Channel-mixing part, which provides within-patch communication.
            x = RDCAB(
                num_channels=features,
                reduction=channels_reduction,
                use_bias=True,
                name=f"{name}_channel_attention_block_1_{i}",
            )(x)

        # long skip-connect
        x = x + shortcut_long
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
    use_bias: bool = True,
    use_cross_gating: bool = False,
    name: str = "unet_encoder",
):
    """Encoder block in MAXIM."""

    def apply(x, skip=None, enc=None, dec=None):
        if skip is not None:
            x = tf.concat([x, skip], axis=-1)

        # convolution-in
        x = layers.Conv2D(filters=num_channels, use_bias=use_bias, kernel_size=(1, 1), padding="same", name=f"{name}_Conv_0")(x)
        shortcut_long = x

        for i in range(num_groups):
            if use_global_mlp:
                x = ResidualSplitHeadMultiAxisGmlpLayer(
                    grid_size=grid_size,
                    block_size=block_size,
                    grid_gmlp_factor=grid_gmlp_factor,
                    block_gmlp_factor=block_gmlp_factor,
                    input_proj_factor=input_proj_factor,
                    use_bias=use_bias,
                    dropout_rate=dropout_rate,
                    name=f"{name}_SplitHeadMultiAxisGmlpLayer_{i}",
                )(x)
            x = RCAB(
                num_channels=num_channels,
                reduction=channels_reduction,
                lrelu_slope=lrelu_slope,
                use_bias=use_bias,
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
                use_bias=use_bias,
                name=f"{name}_cross_gating_block",
            )(x, enc + dec)

        if downsample:
            x_down = layers.Conv2D(
                filters=num_channels, use_bias=use_bias, kernel_size=(4, 4), strides=(2, 2), padding="same", name=f"{name}_Conv_1"
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
    use_bias: bool = True,
    name: str = "unet_decoder",
):

    """Decoder block in MAXIM."""

    def apply(x, bridge=None):
        x = layers.Conv2D(
                filters=num_channels, use_bias=use_bias, kernel_size=(4, 4), strides=(2, 2), padding="same", name=f"{name}_ConvTranspose_0"
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
            use_bias=use_bias,
            name=f"{name}_UNetEncoderBlock_0",
        )(x, skip=bridge)

        return x

    return apply

def UpSampleRatio():
  """Upsample features given a ratio > 0."""
  features: int
  ratio: float
  use_bias: bool = True

  @nn.compact
  def __call__(self, x):
    n, h, w, c = x.shape
    x = jax.image.resize(
        x,
        shape=(n, int(h * self.ratio), int(w * self.ratio), c),
        method="bilinear")
    x = Conv1x1(features=self.features, use_bias=self.use_bias)(x)
    return x


def MAXIM(
    features: int = 64,
    depth: int = 3,
    num_stages: int = 2,
    num_groups: int = 1,
    use_bias: bool = True,
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
                x_scale = layers.Conv2D(
                    filters=(2 ** i) * features,
                    use_bias=use_bias,
                    kernel_size=(3, 3), padding="same",
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
                            use_bias=use_bias,
                            name=f"stage_{idx_stage}_input_fuse_sam_{i}",
                        )(x_scale, sam_features.pop())
                    else:
                        x_scale = layers.Conv2D(
                            filters=(2 ** i) * features,
                            use_bias=use_bias,
                            kernel_size=(1, 1), padding="same",
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
                    use_bias=use_bias,
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
                    use_bias=use_bias,
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
                            use_bias=use_bias,
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
                        use_bias=use_bias,
                        name=f"stage_{idx_stage}_cross_gating_block_{i}",
                    )(signal, global_feature)
                else:
                    skips = layers.Conv2D(
                        filters=(2 ** i) * features, kernel_size=(1, 1), padding="same", use_bias=use_bias, name="Conv_0"
                    )(signal)
                    skips = layers.Conv2D(
                        filters=(2 ** i) * features, kernel_size=(3, 3), padding="same", use_bias=use_bias, name="Conv_1"
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
                            use_bias=use_bias,
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
                    use_bias=use_bias,
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
                            use_bias=use_bias,
                            name=f"stage_{idx_stage}_supervised_attention_module_{i}",
                        )(x, shortcuts[i])
                        outputs.append(output)
                        sam_features.append(sam)
                    else:  # Last stage, apply output convolutions
                        output = layers.Conv2D(
                            num_outputs,
                            use_bias=use_bias,
                            kernel_size=(3, 3), padding="same",
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