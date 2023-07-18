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

from tensorflow import keras
from tensorflow.keras import initializers
from tensorflow.keras import layers
from tensorflow.keras import regularizers


def Block(filters, downsample, sync_bn):
    """A default block which serves as an example of the block interface.

    This is the base block definition for a CenterPillar model.

    Note that the sync_bn parameter is a temporary workaround and should _not_
    be part of the Block API.
    """

    def apply(x):
        input_depth = x.shape.as_list()[-1]
        stride = 2 if downsample else 1

        residual = x

        x = layers.Conv2D(
            filters,
            3,
            stride,
            padding="same",
            use_bias=False,
            kernel_initializer=initializers.VarianceScaling(),
            kernel_regularizer=regularizers.L2(l2=1e-4),
        )(x)
        if sync_bn:
            x = layers.BatchNormalization(
                synchronized=True,
            )(x)
        else:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(
            filters,
            3,
            1,
            padding="same",
            use_bias=False,
            kernel_initializer=initializers.VarianceScaling(),
            kernel_regularizer=regularizers.L2(l2=1e-4),
        )(x)
        if sync_bn:
            x = layers.BatchNormalization(
                synchronized=True,
            )(x)
        else:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        if downsample:
            residual = layers.MaxPool2D(pool_size=2, strides=2, padding="SAME")(
                residual
            )

        if input_depth != filters:
            residual = layers.Conv2D(
                filters,
                1,
                1,
                padding="same",
                use_bias=False,
                kernel_initializer=initializers.VarianceScaling(),
                kernel_regularizer=regularizers.L2(l2=1e-4),
            )(residual)
            if sync_bn:
                residual = layers.BatchNormalization(
                    synchronized=True,
                )(residual)
            else:
                residual = layers.BatchNormalization()(residual)
            residual = layers.ReLU()(residual)

        x = x + residual

        return x

    return apply


def SkipBlock(filters, sync_bn):
    def apply(x):
        x = layers.Conv2D(
            filters,
            1,
            1,
            use_bias=False,
            kernel_initializer=initializers.VarianceScaling(),
            kernel_regularizer=regularizers.L2(l2=1e-4),
        )(x)
        if sync_bn:
            x = layers.BatchNormalization(
                synchronized=True,
            )(x)
        else:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        return x

    return apply


def DownSampleBlock(filters, num_blocks, sync_bn):
    def apply(x):
        x = Block(filters, downsample=True, sync_bn=sync_bn)(x)

        for _ in range(num_blocks - 1):
            x = Block(filters, downsample=False, sync_bn=sync_bn)(x)

        return x

    return apply


def UpSampleBlock(filters, sync_bn):
    def apply(x, lateral_input):
        x = layers.Conv2DTranspose(
            filters,
            3,
            2,
            padding="same",
            use_bias=False,
            kernel_initializer=initializers.VarianceScaling(),
            kernel_regularizer=regularizers.L2(l2=1e-4),
        )(x)
        if sync_bn:
            x = layers.BatchNormalization(
                synchronized=True,
            )(x)
        else:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        lateral_input = SkipBlock(filters, sync_bn=sync_bn)(lateral_input)

        x = x + lateral_input
        x = Block(filters, downsample=False, sync_bn=sync_bn)(x)

        return x

    return apply


def UNet(
    input_shape,
    down_block_configs,
    up_block_configs,
    down_block=DownSampleBlock,
    up_block=UpSampleBlock,
    sync_bn=True,
):
    """Experimental UNet API. This API should not be considered stable.

    All up and down blocks scale by a factor of two. Skip connections are
    included.

    All function parameters require curried functions as inputs which return a
    function that acts on tensors as inputs.

    Reference:
        - [U-Net: Convolutional Networks for Biomedical Image Segmentation]
          (https://arxiv.org/abs/1505.04597)
        
    Args:
        input_shape: the rank 3 shape of the input to the UNet
        down_block_configs: a list of (filter_count, num_blocks) tuples
            indicating the number of filters and sub-blocks in each down block
        up_block_configs: a list of filter counts, one for each up block
        down_block: a downsampling block
        up_block: an upsampling block
    """

    input = layers.Input(shape=input_shape)
    x = input

    skip_connections = []
    # Filters refers to the number of convolutional filters in each block,
    # while num_blocks refers to the number of sub-blocks within a block
    # (Note that only the first sub-block will perform downsampling)
    for filters, num_blocks in down_block_configs:
        skip_connections.append(x)
        x = down_block(filters, num_blocks, sync_bn=sync_bn)(x)

    for filters in up_block_configs:
        x = up_block(filters, sync_bn=sync_bn)(x, skip_connections.pop())

    output = x

    return keras.Model(input, output)
