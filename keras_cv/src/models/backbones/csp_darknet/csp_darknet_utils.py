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

from keras_cv.src.backend import keras


def DarknetConvBlock(
    filters, kernel_size, strides, use_bias=False, activation="silu", name=None
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
        activation: the activation applied after the BatchNorm layer. One of
            "silu", "relu" or "leaky_relu", defaults to "silu".
        name: the prefix for the layer names used in the block.
    """

    if name is None:
        name = f"conv_block{keras.backend.get_uid('conv_block')}"

    model_layers = [
        keras.layers.Conv2D(
            filters,
            kernel_size,
            strides,
            padding="same",
            use_bias=use_bias,
            name=name + "_conv",
        ),
        keras.layers.BatchNormalization(name=name + "_bn"),
    ]

    if activation == "silu":
        model_layers.append(
            keras.layers.Lambda(lambda x: keras.activations.silu(x))
        )
    elif activation == "relu":
        model_layers.append(keras.layers.ReLU())
    elif activation == "leaky_relu":
        model_layers.append(keras.layers.LeakyReLU(0.1))

    return keras.Sequential(model_layers, name=name)


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
        name = f"residual_block{keras.backend.get_uid('residual_block')}"

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
                x = keras.layers.Add(name=f"{name}_out")([residual, x])
            else:
                x = keras.layers.Add(name=f"{name}_add_{i}")([residual, x])

        return x

    return apply


def SpatialPyramidPoolingBottleneck(
    filters,
    hidden_filters=None,
    kernel_sizes=(5, 9, 13),
    activation="silu",
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
        activation: Activation for the conv layers, defaults to "silu".
        name: the prefix for the layer names used in the block.

    Returns:
        a function that takes an input Tensor representing an
        SpatialPyramidPoolingBottleneck.
    """
    if name is None:
        name = f"spp{keras.backend.get_uid('spp')}"

    if hidden_filters is None:
        hidden_filters = filters

    def apply(x):
        x = DarknetConvBlock(
            hidden_filters,
            kernel_size=1,
            strides=1,
            activation=activation,
            name=f"{name}_conv1",
        )(x)
        x = [x]

        for kernel_size in kernel_sizes:
            x.append(
                keras.layers.MaxPooling2D(
                    kernel_size,
                    strides=1,
                    padding="same",
                    name=f"{name}_maxpool_{kernel_size}",
                )(x[0])
            )

        x = keras.layers.Concatenate(name=f"{name}_concat")(x)
        x = DarknetConvBlock(
            filters,
            kernel_size=1,
            strides=1,
            activation=activation,
            name=f"{name}_conv2",
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
        name = f"conv_block{keras.backend.get_uid('conv_block')}"

    model_layers = [
        keras.layers.DepthwiseConv2D(
            kernel_size, strides, padding="same", use_bias=False
        ),
        keras.layers.BatchNormalization(),
    ]

    if activation == "silu":
        model_layers.append(
            keras.layers.Lambda(lambda x: keras.activations.swish(x))
        )
    elif activation == "relu":
        model_layers.append(keras.layers.ReLU())
    elif activation == "leaky_relu":
        model_layers.append(keras.layers.LeakyReLU(0.1))

    model_layers.append(
        DarknetConvBlock(
            filters, kernel_size=1, strides=1, activation=activation
        )
    )

    return keras.Sequential(model_layers, name=name)


@keras.saving.register_keras_serializable(package="keras_cv")
class CrossStagePartial(keras.layers.Layer):
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
        activation: the activation applied after the final layer. One of "silu",
            "relu" or "leaky_relu", defaults to "silu".
    """

    def __init__(
        self,
        filters,
        num_bottlenecks,
        residual=True,
        use_depthwise=False,
        activation="silu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.num_bottlenecks = num_bottlenecks
        self.residual = residual
        self.use_depthwise = use_depthwise
        self.activation = activation

        hidden_channels = filters // 2
        ConvBlock = (
            DarknetConvBlockDepthwise if use_depthwise else DarknetConvBlock
        )

        self.darknet_conv1 = DarknetConvBlock(
            hidden_channels,
            kernel_size=1,
            strides=1,
            activation=activation,
        )

        self.darknet_conv2 = DarknetConvBlock(
            hidden_channels,
            kernel_size=1,
            strides=1,
            activation=activation,
        )

        # repeat bottlenecks num_bottleneck times
        self.bottleneck_convs = []
        for _ in range(num_bottlenecks):
            self.bottleneck_convs.append(
                DarknetConvBlock(
                    hidden_channels,
                    kernel_size=1,
                    strides=1,
                    activation=activation,
                )
            )

            self.bottleneck_convs.append(
                ConvBlock(
                    hidden_channels,
                    kernel_size=3,
                    strides=1,
                    activation=activation,
                )
            )

        if self.residual:
            self.add = keras.layers.Add()
        self.concatenate = keras.layers.Concatenate()

        self.darknet_conv3 = DarknetConvBlock(
            filters, kernel_size=1, strides=1, activation=activation
        )

    def call(self, x):
        x1 = self.darknet_conv1(x)
        x2 = self.darknet_conv2(x)

        for i in range(self.num_bottlenecks):
            residual = x1
            x1 = self.bottleneck_convs[2 * i](x1)
            x1 = self.bottleneck_convs[2 * i + 1](x1)

            if self.residual:
                x1 = self.add([residual, x1])

        x1 = self.concatenate([x1, x2])
        x = self.darknet_conv3(x1)
        return x

    def get_config(self):
        config = {
            "filters": self.filters,
            "num_bottlenecks": self.num_bottlenecks,
            "residual": self.residual,
            "use_depthwise": self.use_depthwise,
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
        return keras.layers.Concatenate(name=name)(
            [
                x[..., ::2, ::2, :],
                x[..., 1::2, ::2, :],
                x[..., ::2, 1::2, :],
                x[..., 1::2, 1::2, :],
            ],
        )

    return apply
