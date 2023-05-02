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

"""CSPDarkNet model utils for KerasCV.
Reference:
  - [YoloV3 Paper](https://arxiv.org/abs/1804.02767)
  - [YoloV3 implementation](https://github.com/ultralytics/yolov3)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers


def apply_darknet_conv_block(
    inputs,
    filters,
    kernel_size,
    strides,
    use_bias=False,
    padding="same",
    use_zero_padding=False,
    batch_norm_momentum=0.99,
    activation="silu",
    name=None,
):
    """The basic conv block used in Darknet. Applies Conv2D followed by a
    BatchNorm.

    Args:
        inputs: The input tensor.
        filters: Integer, the dimensionality of the output space (i.e. the
            number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window. Can be a single
            integer to specify the same value both dimensions.
        strides: An integer or tuple/list of 2 integers, specifying the strides
            of the convolution along the height and width. Can be a single
            integer to the same value both dimensions.
        use_bias: Boolean, whether the layer uses a bias vector.
        padding: String, the padding used in the `Conv2D` layer. Defaults to
            `"same"`.
        use_zero_padding: Boolean, whether to use `ZeroPadding2D` layer at the
            beginning of the block. The zero padding will only be applied when
            `kernel_size` > 1. Defaults to `False`.
        batch_norm_momentum: Float, momentum for the moving average for the
            `BatchNormalization` layer. Defaults to `0.99`.
        activation: the activation applied after the `BatchNormalization` layer.
            One of `"silu"`, `"relu"` or `"leaky_relu"`, defaults to `"silu"`.
        name: the prefix for the layer names used in the block.

    Returns:
        The output tensor.
    """
    x = inputs

    if name is None:
        name = f"conv_block{backend.get_uid('conv_block')}"

    if kernel_size > 1 and use_zero_padding:
        x = layers.ZeroPadding2D(padding=kernel_size // 2, name=f"{name}_pad")(
            x
        )

    x = layers.Conv2D(
        filters,
        kernel_size,
        strides,
        padding=padding,
        use_bias=use_bias,
        name=name + "_conv",
    )(x)
    x = layers.BatchNormalization(
        momentum=batch_norm_momentum, name=f"{name}_bn"
    )(x)

    if activation == "silu":
        x = layers.Activation("swish", name=f"{name}_activation")(x)
    elif activation == "relu":
        x = layers.ReLU()(x)
    elif activation == "leaky_relu":
        x = layers.LeakyReLU(0.1)(x)

    return x


def ResidualBlocks(filters, num_blocks, name=None):
    """A residual block used in DarkNet models, repeated `num_blocks` times.

    Args:
        filters: Integer, the dimensionality of the output spaces (i.e. the
            number of output filters in used the blocks).
        num_blocks: number of times the residual connections are repeated
        name: the prefix for the layer names used in the block.

    Returns:
        a function that takes an input Tensor representing a ResidualBlock.
    """

    if name is None:
        name = f"residual_block{backend.get_uid('residual_block')}"

    def apply(x):
        x = apply_darknet_conv_block(
            x,
            filters,
            kernel_size=3,
            strides=2,
            activation="leaky_relu",
            name=f"{name}_conv1",
        )

        for i in range(1, num_blocks + 1):
            residual = x

            x = apply_darknet_conv_block(
                x,
                filters // 2,
                kernel_size=1,
                strides=1,
                activation="leaky_relu",
                name=f"{name}_conv{2*i}",
            )
            x = apply_darknet_conv_block(
                x,
                filters,
                kernel_size=3,
                strides=1,
                activation="leaky_relu",
                name=f"{name}_conv{2*i + 1}",
            )

            if i == num_blocks:
                x = layers.Add(name=f"{name}_out")([residual, x])
            else:
                x = layers.Add(name=f"{name}_add_{i}")([residual, x])

        return x

    return apply


def apply_spatial_pyramid_pooling_bottleneck(
    inputs,
    filters,
    hidden_filters=None,
    kernel_sizes=(5, 9, 13),
    padding="same",
    use_zero_padding=False,
    batch_norm_momentum=0.99,
    activation="silu",
    sequential_pooling=False,
    name=None,
):
    """Spatial pyramid pooling layer used in YOLOv3-SPP

    Args:
        inputs: The input tensor.
        filters: Integer, the dimensionality of the output spaces (i.e. the
            number of output filters in used the blocks).
        hidden_filters: Integer, the dimensionality of the intermediate
            bottleneck space (i.e. the number of output filters in the
            bottleneck convolution). If None, it will be equal to filters.
            Defaults to None.
        kernel_sizes: A list or tuple representing all the pool sizes used for
            the pooling layers, defaults to (5, 9, 13).
        padding: String, the padding used in the `Conv2D` layers in the
            `DarknetConvBlock`s. Defaults to `"same"`.
        use_zero_padding: Boolean, whether to use `ZeroPadding2D` layer at the
            beginning of each `DarknetConvBlock`. The zero padding will only be
            applied when `kernel_size` > 1. Defaults to `False`.
        batch_norm_momentum: Float, momentum for the moving average for the
            `BatchNormalization` layers in the `DarknetConvBlock`s. Defaults to
            `0.99`.
        activation: Activation for the conv layers, defaults to "silu".
        sequential_pooling: Boolean, whether the `MaxPooling2D` layers are
            applied sequentially. If `True`, the output of a `MaxPooling2D`
            layer will be the input to the next `MaxPooling2D` layer. If
            `False`, the same input tensor is used to feed all the
            `MaxPooling2D` layers. Defaults to `False`.
        name: the prefix for the layer names used in the block.

    Returns:
        The output tensor.
    """
    if name is None:
        name = f"spp{backend.get_uid('spp')}"

    if hidden_filters is None:
        hidden_filters = filters

    x = inputs
    x = apply_darknet_conv_block(
        x,
        hidden_filters,
        kernel_size=1,
        strides=1,
        activation=activation,
        name=f"{name}_conv_1",
        batch_norm_momentum=batch_norm_momentum,
        use_zero_padding=use_zero_padding,
        padding=padding,
    )

    outputs = [x]

    for index, kernel_size in enumerate(kernel_sizes):
        layer = layers.MaxPooling2D(
            pool_size=kernel_size,
            strides=1,
            padding="same",
            name=f"{name}_maxpool_{index}",
        )
        if sequential_pooling:
            output = layer(outputs[-1])
        else:
            output = layer(x)
        outputs.append(output)

    x = layers.Concatenate(name=f"{name}_concat")(outputs)
    x = apply_darknet_conv_block(
        x,
        filters,
        kernel_size=1,
        strides=1,
        activation=activation,
        name=f"{name}_conv_2",
        batch_norm_momentum=batch_norm_momentum,
        use_zero_padding=use_zero_padding,
        padding=padding,
    )

    return x


def apply_darknet_conv_block_depthwise(
    inputs,
    filters,
    kernel_size,
    strides,
    padding="same",
    use_zero_padding=False,
    batch_norm_momentum=0.99,
    activation="silu",
    name=None,
):
    """The depthwise conv block used in CSPDarknet.

    Args:
        inputs: The input tensor.
        filters: Integer, the dimensionality of the output space (i.e. the
            number of output filters in the final convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window. Can be a single
            integer to specify the same value both dimensions.
        strides: An integer or tuple/list of 2 integers, specifying the strides
            of the convolution along the height and width. Can be a single
            integer to the same value both dimensions.
        padding: String, the padding used in the `Conv2D` layer. Defaults to
            `"same"`.
        use_zero_padding: Boolean, whether to use `ZeroPadding2D` layer at the
            beginning of the block. The zero padding will only be applied when
            `kernel_size` > 1. Defaults to `False`.
        batch_norm_momentum: Float, momentum for the moving average for the
            `BatchNormalization` layer. Defaults to `0.99`.
        activation: the activation applied after the final layer. One of "silu",
            "relu" or "leaky_relu", defaults to "silu".
        name: the prefix for the layer names used in the block.

    Returns:
        The output tensor.
    """

    if name is None:
        name = f"conv_block{backend.get_uid('conv_block')}"

    x = inputs
    x = layers.DepthwiseConv2D(
        kernel_size, strides, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)

    if activation == "silu":
        x = layers.Lambda(lambda x: keras.activations.swish(x))(x)
    elif activation == "relu":
        x = layers.ReLU()(x)
    elif activation == "leaky_relu":
        x = layers.LeakyReLU(0.1)(x)

    x = apply_darknet_conv_block(
        x,
        filters,
        kernel_size=1,
        strides=1,
        padding=padding,
        use_zero_padding=use_zero_padding,
        batch_norm_momentum=batch_norm_momentum,
        activation=activation,
    )

    return x


def apply_cross_stage_partial(
    inputs,
    filters,
    num_bottlenecks,
    residual=True,
    use_depthwise=False,
    wide_stem=False,
    kernel_sizes=[1, 3],
    concat_bottleneck_outputs=False,
    padding="same",
    use_zero_padding=False,
    batch_norm_momentum=0.99,
    activation="silu",
    name=None,
):
    """A block used in Cross Stage Partial Darknet.

    Args:
        inputs: The input tensor.
        filters: Integer, the dimensionality of the output space (i.e. the
            number of output filters in the final convolution).
        num_bottlenecks: an integer representing the number of blocks added in
            the layer bottleneck.
        residual: a boolean representing whether the value tensor before the
            bottleneck should be added to the output of the bottleneck as a
            residual, defaults to True.
        use_depthwise: a boolean value used to decide whether a depthwise conv
            block should be used over a regular darknet block, defaults to
            False.
        wide_stem: Boolean, whether to combine the first two `DarknetConvBlock`s
            into one with more filters and split the outputs to two tensors.
            Defaults to `False`.
        kernel_sizes: A list of integers of length 2. The kernel sizes of the
            bottleneck layers. Defaults to `[1, 3]`.
        concat_bottleneck_outputs: Boolean, whether to concatenate the outputs
            of all the bottleneck blocks as the output for the next layer. If
            `False`, only the output of the last bottleneck block is used.
            Defaults to `False`.
        padding: String, the padding used in the `Conv2D` layers in the
            `DarknetConvBlock`s. Defaults to `"same"`.
        use_zero_padding: Boolean, whether to use `ZeroPadding2D` layer at the
            beginning of each `DarknetConvBlock`. The zero padding will only be
            applied when `kernel_size` > 1. Defaults to `False`.
        batch_norm_momentum: Float, momentum for the moving average for the
            `BatchNormalization` layers in the `DarknetConvBlock`s. Defaults to
            `0.99`.
        activation: the activation applied after the final layer. One of "silu",
            "relu" or "leaky_relu", defaults to "silu".

    Returns:
        The output tensor.
    """
    x = inputs
    hidden_channels = filters // 2
    apply_conv_block = (
        apply_darknet_conv_block_depthwise
        if use_depthwise
        else apply_darknet_conv_block
    )

    if wide_stem:
        pre = apply_darknet_conv_block(
            x,
            hidden_channels * 2,
            kernel_size=1,
            strides=1,
            activation=activation,
            batch_norm_momentum=batch_norm_momentum,
            use_zero_padding=use_zero_padding,
            padding=padding,
            name=f"{name}_conv_1",
        )
        short, deep = tf.split(pre, 2, axis=-1)
    else:
        deep = apply_darknet_conv_block(
            x,
            hidden_channels,
            kernel_size=1,
            strides=1,
            activation=activation,
            batch_norm_momentum=batch_norm_momentum,
            use_zero_padding=use_zero_padding,
            padding=padding,
            name=f"{name}_conv_1",
        )

        short = apply_darknet_conv_block(
            x,
            hidden_channels,
            kernel_size=1,
            strides=1,
            activation=activation,
            batch_norm_momentum=batch_norm_momentum,
            use_zero_padding=use_zero_padding,
            padding=padding,
            name=f"{name}_conv_2",
        )

    out = [short, deep]
    add = layers.Add(name=f"{name}_add")
    for index in range(num_bottlenecks):
        deep = apply_darknet_conv_block(
            deep,
            hidden_channels,
            kernel_size=kernel_sizes[0],
            strides=1,
            activation=activation,
            batch_norm_momentum=batch_norm_momentum,
            use_zero_padding=use_zero_padding,
            padding=padding,
            name=f"{name}_bottleneck_{index}_1",
        )
        deep = apply_conv_block(
            deep,
            hidden_channels,
            kernel_size=kernel_sizes[1],
            strides=1,
            activation=activation,
            batch_norm_momentum=batch_norm_momentum,
            use_zero_padding=use_zero_padding,
            padding=padding,
            name=f"{name}_bottleneck_{index}_2",
        )

        if residual:
            deep = add([out[-1], deep])
        out.append(deep)

    concatenate = layers.Concatenate(name=f"{name}_concat")
    if concat_bottleneck_outputs:
        x = concatenate(out)
    else:
        x = concatenate([deep, short])
    x = apply_darknet_conv_block(
        x,
        filters,
        kernel_size=1,
        strides=1,
        activation=activation,
        batch_norm_momentum=batch_norm_momentum,
        use_zero_padding=use_zero_padding,
        padding=padding,
        name=f"{name}_conv_3",
    )
    return x


def apply_focus(inputs, name=None):
    """A block used in CSPDarknet to focus information into channels of the
    image.

    If the dimensions of a batch input is (batch_size, width, height, channels),
    this layer converts the image into size (batch_size, width/2, height/2,
    4*channels). See [the original discussion on YoloV5 Focus Layer](https://github.com/ultralytics/yolov5/discussions/3181).

    Args:
        inputs: The input tensor.
        name: the name for the lambda layer used in the block.

    Returns:
        The output tensor.
    """  # noqa: E501

    x = inputs
    return layers.Lambda(
        lambda x: tf.concat(
            [
                x[..., ::2, ::2, :],
                x[..., 1::2, ::2, :],
                x[..., ::2, 1::2, :],
                x[..., 1::2, 1::2, :],
            ],
            axis=-1,
        ),
        name=name,
    )(x)
