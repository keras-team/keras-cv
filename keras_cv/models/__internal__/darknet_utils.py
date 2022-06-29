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

"""DarkNet model utils for KerasCV.
Reference:
  - [YoloV3 Paper](https://arxiv.org/abs/1804.02767)
  - [YoloV3 implementation](https://github.com/ultralytics/yolov3)
"""

from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers


def DarknetConvBlock(
    filters,
    kernel_size,
    strides,
    groups=1,
    use_bias=False,
    activation="silu",
    name=None,
):
    """The basic conv block used in Darknet. Applies Conv2D followed by a BatchNorm.

    Args:
        filters: Integer, the dimensionality of the output space (i.e. the number of
            output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the height
            and width of the 2D convolution window. Can be a single integer to specify
            the same value both dimensions.
        strides: An integer or tuple/list of 2 integers, specifying the strides of
            the convolution along the height and width. Can be a single integer to
            the same value both dimensions.
        groups: A positive integer specifying the number of groups in which the
            input is split along the channel axis. Each group is convolved separately
            with `filters / groups` filters. The output is the concatenation of all
            the `groups` results along the channel axis. Input channels and `filters`
            must both be divisible by `groups`.
        use_bias: Boolean, whether the layer uses a bias vector.
        activation: the activation applied after the BatchNorm layer. One of "silu",
            "relu" or "lrelu". Defaults to "silu".
        name: the prefix for the layer names used in the block.

    Returns:
        a function that takes an input Tensor representing a DarknetConvBlock.
    """

    if name is None:
        name = f"darknet_block{backend.get_uid('darknet_block')}"

    def apply(x):
        x = layers.Conv2D(
            filters,
            kernel_size,
            strides,
            padding="same",
            groups=groups,
            use_bias=use_bias,
            name=name,
        )(x)
        x = layers.BatchNormalization(name=f"{name}_bn")(x)

        if activation == "silu":
            x = layers.Lambda(
                lambda x: keras.activations.swish(x), name=f"{name}_silu"
            )(x)
        elif activation == "relu":
            x = layers.ReLU(name=f"{name}_relu")(x)
        elif activation == "lrelu":
            x = layers.LeakyReLU(0.1, name=f"{name}_lrelu")(x)

        return x

    return apply


def ResidualBlocks(filters, num_blocks, name=None):
    """A residual block used in DarkNet models, repeated `num_blocks` times.

    Args:
        filters: Integer, the dimensionality of the output spaces (i.e. the number of
            output filters in used the blocks).
        num_blocks: number of times the residual connections are repeated
        name: the prefix for the layer names used in the block.

    Returns:
        a function that takes an input Tensor representing a ResidualBlock.
    """

    if name is None:
        name = f"residual_block{backend.get_uid('residual_block')}"

    def apply(x):
        x = DarknetConvBlock(
            filters, kernel_size=3, strides=2, activation="lrelu", name=f"{name}_conv1"
        )(x)

        for i in range(1, num_blocks + 1):
            residual = x

            x = DarknetConvBlock(
                filters // 2,
                kernel_size=1,
                strides=1,
                activation="lrelu",
                name=f"{name}_conv{2*i}",
            )(x)
            x = DarknetConvBlock(
                filters,
                kernel_size=3,
                strides=1,
                activation="lrelu",
                name=f"{name}_conv{2*i + 1}",
            )(x)

            if i == num_blocks:
                x = layers.Add(name=f"{name}_out")([residual, x])
            else:
                x = layers.Add(name=f"{name}_add_{i}")([residual, x])

        return x

    return apply


def SPPBottleneck(filters, kernel_sizes=(5, 9, 13), activation="silu", name=None):
    """Spatial pyramid pooling layer used in YOLOv3-SPP

    Args:
        filters: Integer, the dimensionality of the output spaces (i.e. the number of
            output filters in used the blocks).
        kernel_sizes: A list or tuple representing all the pool sizes used for the
            pooling layers. Defaults to (5, 9, 13).
        activation: Activation for the conv layers. Defaults to "silu".
        name: the prefix for the layer names used in the block.

    Returns:
        a function that takes an input Tensor representing an SPPBottleneck.
    """

    if name is None:
        name = f"spp{backend.get_uid('spp')}"

    def apply(x):
        x = DarknetConvBlock(
            filters,
            kernel_size=1,
            strides=1,
            activation=activation,
            name=f"{name}_conv1",
        )(x)
        x = [x]

        for kernel_size in kernel_sizes:
            x.append(
                layers.MaxPooling2D(
                    kernel_size,
                    strides=1,
                    padding="same",
                    name=f"{name}_maxpool_{kernel_size}",
                )(x[0])
            )

        x = layers.Concatenate(name=f"{name}_concat")(x)
        x = DarknetConvBlock(
            filters,
            kernel_size=1,
            strides=1,
            activation=activation,
            name=f"{name}_conv2",
        )(x)

        return x

    return apply
