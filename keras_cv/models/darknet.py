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

"""DarkNet models for KerasCV.
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
        name = f"darknet_block_{backend.get_uid('darknet_block')}"

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
            x = keras.activations.swish(x)
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
        name = f"residual_block_{backend.get_uid('residual_block')}"

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
        name = f"spp_{backend.get_uid('spp')}"

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


def DarkNet(
    blocks,
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(None, None, 3),
    pooling=None,
    classifier_activation="softmax",
    name=None,
    **kwargs,
):
    """Instantiates the DarkNet architecture.

    Although the DarkNet architecture is commonly used for detection tasks, it is
    possible to extract the intermediate dark2 to dark5 layers from the model for
    creating a feature pyramid Network.

    Reference:
        - [YoloV3 Paper](https://arxiv.org/abs/1804.02767)
        - [YoloV3 implementation](https://github.com/ultralytics/yolov3)
    For transfer learning use cases, make sure to read the [guide to transfer learning
    & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        blocks: numbers of building blocks from the layer dark2 to layer dark5.
        include_rescaling: whether or not to Rescale the inputs.If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        include_top: whether to include the fully-connected
            layer at the top of the network.  If provided, `classes` must be
            provided.
        classes: optional number of classes to classify imagesinto, only to be
            specified if `include_top` is True, and if no `weights` argument is
            specified.
        weights: one of `None` (random initialization), or a pretrained weight
            file path.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        pooling: optional pooling mode for feature extraction when `include_top`
            is `False`.
            - `None` means that the output of the model will be the 4D tensor output
                of the last convolutional block.
            - `avg` means that global average pooling will be applied to the
                output of the last convolutional block, and thus the output of the
                model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.
            When loading pretrained weights, `classifier_activation` can only
            be `None` or `"softmax"`.
        name: (Optional) name to pass to the model.  Defaults to "DarkNet".
    Returns:
        A `keras.Model` instance.
    """
    if weights and not tf.io.gfile.exists(weights):
        raise ValueError(
            "The `weights` argument should be either `None` or the path to the "
            f"weights file to be loaded. Weights file not found at location: {weights}"
        )

    if include_top and not classes:
        raise ValueError(
            "If `include_top` is True, you should specify `classes`. Received: "
            f"classes={classes}"
        )

    inputs = layers.Input(shape=input_shape)

    x = inputs
    if include_rescaling:
        x = layers.Rescaling(1 / 255.0)(x)

    # stem
    x = DarknetConvBlock(
        filters=32, kernel_size=3, strides=1, activation="lrelu", name="stem_conv"
    )(x)
    x = ResidualBlocks(filters=64, num_blocks=1, name="stem_residual_block")(x)

    # dark2
    x = ResidualBlocks(filters=128, num_blocks=blocks[0], name="dark2_residual_block")(
        x
    )

    # dark3
    x = ResidualBlocks(filters=256, num_blocks=blocks[1], name="dark3_residual_block")(
        x
    )

    # dark4
    x = ResidualBlocks(filters=512, num_blocks=blocks[2], name="dark4_residual_block")(
        x
    )

    # dark5
    x = ResidualBlocks(filters=1024, num_blocks=blocks[3], name="dark5_residual_block")(
        x
    )
    x = DarknetConvBlock(
        filters=512, kernel_size=1, strides=1, activation="lrelu", name="dark5_conv1"
    )(x)
    x = DarknetConvBlock(
        filters=1024, kernel_size=3, strides=1, activation="lrelu", name="dark5_conv2"
    )(x)
    x = SPPBottleneck(512, activation="lrelu", name="dark5_spp")(x)
    x = DarknetConvBlock(
        filters=1024, kernel_size=3, strides=1, activation="lrelu", name="dark5_conv3"
    )(x)
    x = DarknetConvBlock(
        filters=512, kernel_size=1, strides=1, activation="lrelu", name="dark5_conv4"
    )(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = layers.Dense(classes, activation=classifier_activation, name="predictions")(
            x
        )
    elif pooling == "avg":
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    elif pooling == "max":
        x = layers.GlobalMaxPooling2D(name="max_pool")(x)

    model = keras.Model(inputs, x, name=name, **kwargs)

    if weights is not None:
        model.load_weights(weights)
    return model


def DarkNet21(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(None, None, 3),
    pooling=None,
    name="darknet21",
    **kwargs,
):
    return DarkNet(
        [1, 2, 2, 1],
        include_rescaling=include_rescaling,
        include_top=include_top,
        classes=classes,
        weights=weights,
        input_shape=input_shape,
        pooling=pooling,
        name=name,
        **kwargs,
    )


def DarkNet53(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(None, None, 3),
    pooling=None,
    name="darknet53",
    **kwargs,
):
    return DarkNet(
        [2, 8, 8, 4],
        include_rescaling=include_rescaling,
        include_top=include_top,
        classes=classes,
        weights=weights,
        input_shape=input_shape,
        pooling=pooling,
        name=name,
        **kwargs,
    )


setattr(DarkNet21, "__doc__", DarkNet.__doc__)
setattr(DarkNet53, "__doc__", DarkNet.__doc__)
