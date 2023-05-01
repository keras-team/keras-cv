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


def DarknetConvBlock(
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
    """

    if name is None:
        name = f"conv_block{backend.get_uid('conv_block')}"

    model_layers = []
    if kernel_size > 1 and use_zero_padding:
        model_layers.append(
            layers.ZeroPadding2D(padding=kernel_size // 2, name=f"{name}_pad")
        )

    model_layers.append(
        layers.Conv2D(
            filters,
            kernel_size,
            strides,
            padding=padding,
            use_bias=use_bias,
            name=name + "_conv",
        )
    )
    model_layers.append(
        layers.BatchNormalization(
            momentum=batch_norm_momentum, name=name + "_bn"
        ),
    )

    if activation == "silu":
        model_layers.append(
            layers.Activation("swish", name=name + "_activation")
        )
    elif activation == "relu":
        model_layers.append(layers.ReLU())
    elif activation == "leaky_relu":
        model_layers.append(layers.LeakyReLU(0.1))

    return keras.Sequential(model_layers, name=None)


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
        x = DarknetConvBlock(
            filters,
            kernel_size=3,
            strides=2,
            activation="leaky_relu",
            name=f"{name}_conv1",
        )(x)

        for i in range(1, num_blocks + 1):
            residual = x

            x = DarknetConvBlock(
                filters // 2,
                kernel_size=1,
                strides=1,
                activation="leaky_relu",
                name=f"{name}_conv{2*i}",
            )(x)
            x = DarknetConvBlock(
                filters,
                kernel_size=3,
                strides=1,
                activation="leaky_relu",
                name=f"{name}_conv{2*i + 1}",
            )(x)

            if i == num_blocks:
                x = layers.Add(name=f"{name}_out")([residual, x])
            else:
                x = layers.Add(name=f"{name}_add_{i}")([residual, x])

        return x

    return apply


def SpatialPyramidPoolingBottleneck(
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
        a function that takes an input Tensor representing an
        SpatialPyramidPoolingBottleneck.
    """
    if name is None:
        name = f"spp{backend.get_uid('spp')}"

    if hidden_filters is None:
        hidden_filters = filters

    def apply(x):
        x = DarknetConvBlock(
            hidden_filters,
            kernel_size=1,
            strides=1,
            activation=activation,
            name=f"{name}_conv_1",
            batch_norm_momentum=batch_norm_momentum,
            use_zero_padding=use_zero_padding,
            padding=padding,
        )(x)

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
        x = DarknetConvBlock(
            filters,
            kernel_size=1,
            strides=1,
            activation=activation,
            name=f"{name}_conv_2",
            batch_norm_momentum=batch_norm_momentum,
            use_zero_padding=use_zero_padding,
            padding=padding,
        )(x)

        return x

    return apply


def DarknetConvBlockDepthwise(
    filters, kernel_size, strides, activation="silu", name=None
):
    """The depthwise conv block used in CSPDarknet.

    Args:
        filters: Integer, the dimensionality of the output space (i.e. the
            number of output filters in the final convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window. Can be a single
            integer to specify the same value both dimensions.
        strides: An integer or tuple/list of 2 integers, specifying the strides
            of the convolution along the height and width. Can be a single
            integer to the same value both dimensions.
        activation: the activation applied after the final layer. One of "silu",
            "relu" or "leaky_relu", defaults to "silu".
        name: the prefix for the layer names used in the block.

    """

    if name is None:
        name = f"conv_block{backend.get_uid('conv_block')}"

    model_layers = [
        layers.DepthwiseConv2D(
            kernel_size, strides, padding="same", use_bias=False
        ),
        layers.BatchNormalization(),
    ]

    if activation == "silu":
        model_layers.append(layers.Lambda(lambda x: keras.activations.swish(x)))
    elif activation == "relu":
        model_layers.append(layers.ReLU())
    elif activation == "leaky_relu":
        model_layers.append(layers.LeakyReLU(0.1))

    model_layers.append(
        DarknetConvBlock(
            filters, kernel_size=1, strides=1, activation=activation
        )
    )

    return keras.Sequential(model_layers, name=name)


@keras.utils.register_keras_serializable(package="keras_cv")
class CrossStagePartial(keras.Model):
    """A block used in Cross Stage Partial Darknet.

    Args:
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
    """

    def __init__(
        self,
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
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.num_bottlenecks = num_bottlenecks
        self.residual = residual
        self.use_depthwise = use_depthwise
        self.wide_stem = wide_stem
        self.kernel_sizes = kernel_sizes
        self.concat_bottleneck_outputs = concat_bottleneck_outputs
        self.padding = padding
        self.use_zero_padding = use_zero_padding
        self.batch_norm_momentum = batch_norm_momentum
        self.activation = activation

        hidden_channels = filters // 2
        ConvBlock = (
            DarknetConvBlockDepthwise if use_depthwise else DarknetConvBlock
        )

        if self.wide_stem:
            self.darknet_conv1 = DarknetConvBlock(
                hidden_channels * 2,
                kernel_size=1,
                strides=1,
                activation=activation,
                batch_norm_momentum=batch_norm_momentum,
                use_zero_padding=use_zero_padding,
                padding=padding,
                name=f"{name}_conv_1",
            )
        else:
            self.darknet_conv1 = DarknetConvBlock(
                hidden_channels,
                kernel_size=1,
                strides=1,
                activation=activation,
                batch_norm_momentum=batch_norm_momentum,
                use_zero_padding=use_zero_padding,
                padding=padding,
                name=f"{name}_conv_1",
            )

            self.darknet_conv2 = DarknetConvBlock(
                hidden_channels,
                kernel_size=1,
                strides=1,
                activation=activation,
                batch_norm_momentum=batch_norm_momentum,
                use_zero_padding=use_zero_padding,
                padding=padding,
                name=f"{name}_conv_2",
            )

        # repeat bottlenecks num_bottleneck times
        self.bottleneck_convs_1 = []
        self.bottleneck_convs_2 = []
        for index in range(num_bottlenecks):
            self.bottleneck_convs_1.append(
                DarknetConvBlock(
                    hidden_channels,
                    kernel_size=self.kernel_sizes[0],
                    strides=1,
                    activation=activation,
                    batch_norm_momentum=batch_norm_momentum,
                    use_zero_padding=use_zero_padding,
                    padding=padding,
                    name=f"{name}_bottleneck_{index}_1",
                )
            )

            self.bottleneck_convs_2.append(
                ConvBlock(
                    hidden_channels,
                    kernel_size=self.kernel_sizes[1],
                    strides=1,
                    activation=activation,
                    batch_norm_momentum=batch_norm_momentum,
                    use_zero_padding=use_zero_padding,
                    padding=padding,
                    name=f"{name}_bottleneck_{index}_2",
                )
            )

        self.add = layers.Add(name=f"{name}_add")
        self.concatenate = layers.Concatenate(name=f"{name}_concat")

        self.darknet_conv3 = DarknetConvBlock(
            filters,
            kernel_size=1,
            strides=1,
            activation=activation,
            batch_norm_momentum=batch_norm_momentum,
            use_zero_padding=use_zero_padding,
            padding=padding,
            name=f"{name}_conv_3",
        )

    def call(self, x):
        if self.wide_stem:
            pre = self.darknet_conv1(x)
            short, deep = tf.split(pre, 2, axis=-1)
        else:
            deep = self.darknet_conv1(x)
            short = self.darknet_conv2(x)

        out = [short, deep]
        for i in range(self.num_bottlenecks):
            deep = self.bottleneck_convs_1[i](deep)
            deep = self.bottleneck_convs_2[i](deep)

            if self.residual:
                deep = self.add([out[-1], deep])
            out.append(deep)

        if self.concat_bottleneck_outputs:
            x = self.concatenate(out)
        else:
            x = self.concatenate([deep, short])
        x = self.darknet_conv3(x)
        return x

    def get_config(self):
        config = {
            "filters": self.filters,
            "num_bottlenecks": self.num_bottlenecks,
            "residual": self.residual,
            "use_depthwise": self.use_depthwise,
            "wide_stem": self.wide_stem,
            "kernel_sizes": self.kernel_sizes,
            "concat_bottleneck_outputs": self.concat_bottleneck_outputs,
            "padding": self.padding,
            "use_zero_padding": self.use_zero_padding,
            "batch_norm_momentum": self.batch_norm_momentum,
            "activation": self.activation,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def Focus(name=None):
    """A block used in CSPDarknet to focus information into channels of the
    image.

    If the dimensions of a batch input is (batch_size, width, height, channels),
    this layer converts the image into size (batch_size, width/2, height/2,
    4*channels). See [the original discussion on YoloV5 Focus Layer](https://github.com/ultralytics/yolov5/discussions/3181).

    Args:
        name: the name for the lambda layer used in the block.

    Returns:
        a function that takes an input Tensor representing a Focus layer.
    """  # noqa: E501

    def apply(x):
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

    return apply
