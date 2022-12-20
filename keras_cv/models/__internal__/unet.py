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

from tensorflow.keras import initializers
from tensorflow.keras import layers
from tensorflow.keras import regularizers


def Block(filters, downsample):
    def apply(x):
        input_depth = x.shape.as_list()[-1]
        stride = 2 if downsample else 1

        residual = x

        x = layers.Conv2D(
            filters,
            3,
            stride,
            padding="same",
            kernel_initializer=initializers.VarianceScaling(),
            kernel_regularizer=regularizers.L2(l2=1e-4),
        )(x)
        x = layers.BatchNormalization(
            synchronized=True,
            beta_regularizer=regularizers.L2(l2=1e-8),
            gamma_regularizer=regularizers.L2(l2=1e-8),
        )(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(
            filters,
            3,
            1,
            padding="same",
            kernel_initializer=initializers.VarianceScaling(),
            kernel_regularizer=regularizers.L2(l2=1e-4),
        )(x)
        x = layers.BatchNormalization(
            synchronized=True,
            beta_regularizer=regularizers.L2(l2=1e-8),
            gamma_regularizer=regularizers.L2(l2=1e-8),
        )(x)

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
                kernel_initializer=initializers.VarianceScaling(),
                kernel_regularizer=regularizers.L2(l2=1e-4),
            )(residual)

        x = x + residual
        x = layers.ReLU()(x)

        return x

    return apply


def SkipBlock(filters):
    def apply(x):
        x = layers.Conv2D(
            filters,
            1,
            1,
            kernel_initializer=initializers.VarianceScaling(),
            kernel_regularizer=regularizers.L2(l2=1e-4),
        )(x)
        x = layers.BatchNormalization(
            synchronized=True,
            beta_regularizer=regularizers.L2(l2=1e-8),
            gamma_regularizer=regularizers.L2(l2=1e-8),
        )(x)
        x = layers.ReLU()(x)

        return x

    return apply


def DownSampleBlock(filters, num_blocks):
    def apply(x):
        x = Block(filters, downsample=True)(x)

        for _ in range(num_blocks - 1):
            x = Block(filters, downsample=False)(x)

        return x

    return apply


def UpSampleBlock(filters):
    def apply(x, lateral_input):
        x = layers.Conv2DTranspose(
            filters,
            3,
            2,
            padding="same",
            kernel_initializer=initializers.VarianceScaling(),
            kernel_regularizer=regularizers.L2(l2=1e-4),
        )(x)
        x = layers.BatchNormalization(
            synchronized=True,
            beta_regularizer=regularizers.L2(l2=1e-8),
            gamma_regularizer=regularizers.L2(l2=1e-8),
        )(x)
        x = layers.ReLU()(x)

        lateral_input = SkipBlock(filters)(lateral_input)

        x = x + lateral_input
        x = Block(filters, downsample=False)(x)

        return x

    return apply


def UNet(
    down_blocks,
    up_blocks,
    down_block=DownSampleBlock,
    up_block=UpSampleBlock,
):
    """Experimental UNet API. This API should not be considered stable.

    All up and down blocks scale by a factor of two. Skip connections are
    included.

    All function parameters require curried functions as inputs which return a
    function that acts on tensors as inputs.

    Args:
        down_blocks: a list of (filter_count, num_blocks) tuples indicating the
            number of filters and sub-blocks in each down block
        up_blocks: a list of filter counts, one for each up block
        down_block: a downsampling block
        up_block: an upsampling block
    """

    def apply(x):
        skip_connections = []
        for filters, num_blocks in down_blocks:
            skip_connections.append(x)
            x = down_block(filters, num_blocks)(x)

        for filters in up_blocks:
            x = up_block(filters)(x, skip_connections.pop())

        return x

    return apply
