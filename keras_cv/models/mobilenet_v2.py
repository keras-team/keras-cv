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

"""MobileNet v2 models for Keras.

MobileNetV2 is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and different
width factors. This allows different width models to reduce the number of multiply-adds
and thereby reduce inference cost on mobile devices.

References:
    - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](
        https://arxiv.org/abs/1801.04381) (CVPR 2018)
    - [Based on the Original keras.applications MobileNetv2](
        https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet_v2.py)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers


def CorrectPad(kernel_size, name=None):
    """Zero-padding for 2D convolution with downsampling.

    Args:
        kernel_size: an integer or tuple/list of 2 integers.
        name: string, layer label.

    Returns:
        a function that takes an input Tensor representing a CorrectPad.
    """
    if name is None:
        name = f"correct_pad_{backend.get_uid('correct_pad')}"

    def apply(x):
        img_dim = 1
        nonlocal kernel_size
        input_size = backend.int_shape(x)[img_dim : (img_dim + 2)]

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if input_size[0] is None:
            adjust = (1, 1)
        else:
            adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
        correct = (kernel_size[0] // 2, kernel_size[1] // 2)
        return (
            (correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]),
        )

    return apply


def Depth(divisor=8, min_value=None, name=None):
    """Ensure that all layers have a channel number that is divisble by the `divisor`.

    Args:
        divisor: integer, the value by which a channel number should be divisble,
            defaults to 8.
        min_value: float, minimum value for the new tensor.
        name: string, layer label.

    Returns:
        a function that takes an input Tensor representing a Depth layer.
    """
    if name is None:
        name = f"depth_{backend.get_uid('depth')}"

    if min_value is None:
        min_value = divisor

    def apply(x):
        new_x = max(min_value, int(x + divisor / 2) // divisor * divisor)

        # Make sure that round down does not go down by more than 10%.
        if new_x < 0.9 * x:
            new_x += divisor
        return new_x

    return apply


def InvertedResBlock(expansion, stride, alpha, filters, block_id, name=None):

    """An Inverted Residual Block.
    
    Args:
        expansion: integer, the expansion ratio, multiplied with `filters` to get the
            minimum value passed to Depth.
        stride: integer, the stride length for DpethWise Convolutions.
        alpha: float, scaling for the pointwise convolution filters.
        filters: int, number of filters.
        block_id: integer, a unique identification if you want to use expanded
            convolutions.
        name: string, layer label.

    Returns:
        a function that takes an input Tensor representing a InvertedResBlock.    
    """
    if name is None:
        name = f"inverted_res_block_{backend.get_uid('inverted_res_block')}"

    prefix = "block_{}_".format(block_id)

    pointwise_conv_filters = int(filters * alpha)
    # Ensure the number of filters on the last 1x1 convolution is divisible by
    # 8.
    pointwise_filters = Depth(8)(pointwise_conv_filters)

    batch_norm_1 = layers.BatchNormalization(
        axis=-1,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "expand_BN",
    )
    correct_pad = CorrectPad(3)
    activation_1 = layers.ReLU(6.0, name=prefix + "expand_relu")
    depthwise_conv2d_1 = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        activation=None,
        use_bias=False,
        padding="same" if stride == 1 else "valid",
        name=prefix + "depthwise",
    )
    batch_norm_2 = layers.BatchNormalization(
        axis=-1,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "depthwise_BN",
    )
    activation_2 = layers.ReLU(6.0, name=prefix + "depthwise_relu")
    conv2d_1 = layers.Conv2D(
        pointwise_filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        activation=None,
        name=prefix + "project",
    )
    batch_norm_3 = layers.BatchNormalization(
        axis=-1,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "project_BN",
    )
    add = layers.Add(name=prefix + "add")

    def apply(inputs):
        in_channels = backend.int_shape(inputs)[-1]
        prefix = "block_{}_".format(block_id)

        x = inputs

        if block_id:
            # Expand with a pointwise 1x1 convolution.
            x = layers.Conv2D(
                expansion * in_channels,
                kernel_size=1,
                padding="same",
                use_bias=False,
                activation=None,
                name=prefix + "expand",
            )(x)
            x = batch_norm_1(x)
            x = activation_1(x)
        else:
            prefix = "expanded_conv_"

        # Depthwise 3x3 convolution.
        if stride == 2:
            x = layers.ZeroPadding2D(padding=correct_pad(x), name=prefix + "pad")(x)

        x = depthwise_conv2d_1(x)
        x = batch_norm_2(x)

        x = activation_2(x)

        # Project with a pointwise 1x1 convolution.
        x = conv2d_1(x)
        x = batch_norm_3(x)

        if in_channels == pointwise_filters and stride == 1:
            return add([inputs, x])

        return x

    return apply

def MobileNetV2(input_shape=(None, None, 3),
    alpha=1.0,
    include_rescaling=True,
    include_top=True,
    weights=None,
    pooling=None,
    num_classes=None,
    classifier_activation="softmax",
    name="MobileNetV2",
    **kwargs):

    channel_axis = -1

    if weights and not tf.io.gfile.exists(weights):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` or the path to the weights file to be loaded. "
            f"Weights file not found at location: {weights}"
        )

    if include_top and not num_classes:
        raise ValueError(
            "If `include_top` is True, "
            "you should specify `num_classes`. "
            f"Received: num_classes={num_classes}"
        )

    input = layers.Input(shape=input_shape)
    x = None

    if include_rescaling:
        x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(input)

    first_block_filters = Depth(8)(32 * alpha)

    if x is None:
        x = layers.Conv2D(
            first_block_filters,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            use_bias=False,
            name="Conv1",
        )(input)
    else:
        x = layers.Conv2D(
            first_block_filters,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            use_bias=False,
            name="Conv1",
        )(x)

    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name="bn_Conv1"
    )(x)
    x = layers.ReLU(6.0, name="Conv1_relu")(x)

    x = InvertedResBlock(
        filters=16, alpha=alpha, stride=1, expansion=1, block_id=0
    )(x)

    x = InvertedResBlock(
        filters=24, alpha=alpha, stride=2, expansion=6, block_id=1
    )(x)
    x = InvertedResBlock(
        filters=24, alpha=alpha, stride=1, expansion=6, block_id=2
    )(x)

    x = InvertedResBlock(
        filters=32, alpha=alpha, stride=2, expansion=6, block_id=3
    )(x)
    x = InvertedResBlock(
        filters=32, alpha=alpha, stride=1, expansion=6, block_id=4
    )(x)
    x = InvertedResBlock(
        filters=32, alpha=alpha, stride=1, expansion=6, block_id=5
    )(x)

    x = InvertedResBlock(
        filters=64, alpha=alpha, stride=2, expansion=6, block_id=6
    )(x)
    x = InvertedResBlock(
        filters=64, alpha=alpha, stride=1, expansion=6, block_id=7
    )(x)
    x = InvertedResBlock(
        filters=64, alpha=alpha, stride=1, expansion=6, block_id=8
    )(x)
    x = InvertedResBlock(
        filters=64, alpha=alpha, stride=1, expansion=6, block_id=9
    )(x)

    x = InvertedResBlock(
        filters=96, alpha=alpha, stride=1, expansion=6, block_id=10
    )(x)
    x = InvertedResBlock(
        filters=96, alpha=alpha, stride=1, expansion=6, block_id=11
    )(x)
    x = InvertedResBlock(
        filters=96, alpha=alpha, stride=1, expansion=6, block_id=12
    )(x)

    x = InvertedResBlock(
        filters=160, alpha=alpha, stride=2, expansion=6, block_id=13
    )(x)
    x = InvertedResBlock(
        filters=160, alpha=alpha, stride=1, expansion=6, block_id=14
    )(x)
    x = InvertedResBlock(
        filters=160, alpha=alpha, stride=1, expansion=6, block_id=15
    )(x)

    x = InvertedResBlock(
        filters=320, alpha=alpha, stride=1, expansion=6, block_id=16
    )(x)

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we increase the number of output
    # channels.
    if alpha > 1.0:
        last_block_filters = Depth(8)(1280 * alpha)
    else:
        last_block_filters = 1280
    
    x = layers.Conv2D(
        last_block_filters, kernel_size=1, use_bias=False, name="Conv_1"
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name="Conv_1_bn"
    )(x)
    x = layers.ReLU(6.0, name="out_relu")(x)

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(x)
    elif pooling == "avg":
        x = layers.GlobalAveragePooling2D()(x)
    elif pooling == "max":
        x = layers.GlobalMaxPooling2D()(x)

    model = keras.Model(input, x, name=name, **kwargs)

    if weights is not None:
        model.load_weights(weights)

    return model