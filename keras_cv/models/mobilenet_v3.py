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
"""MobileNet v3 models for KerasCV.

Reference:
  - [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf) (ICCV 2019)
  - [Based on the Original keras.applications MobileNetv3](https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet_v3.py)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers

channel_axis = -1


def Depth(divisor=8, min_value=None, name=None):
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


def HardSigmoid(name=None):
    if name is None:
        name = f"hard_sigmoid_{backend.get_uid('hard_sigmoid')}"
        activation = layers.ReLU(6.0)

    def apply(x):
        return activation(x + 3.0) * (1.0 / 6.0)

    return apply


def HardSwish(name=None):
    if name is None:
        name = f"hard_swish_{backend.get_uid('hard_swish')}"
    hard_sigmoid = HardSigmoid()
    multiply_layer = layers.Multiply()

    def apply(x):
        return multiply_layer([x, hard_sigmoid(x)])

    return apply


def SqueezeAndExcitationBlock(filters, se_ratio, prefix, name=None):
    if name is None:
        name = f"se_block_{backend.get_uid('se_block')}"

    if se_ratio <= 0.0 or se_ratio >= 1.0:
        raise ValueError(
            f"`ratio` should be a float between 0 and 1. Got " f" {se_ratio}"
        )

    if filters <= 0 or not isinstance(filters, int):
        raise ValueError(f"`filters` should be a positive integer. Got " f" {filters}")

    ga_pool = layers.GlobalAveragePooling2D(
        keepdims=True, name=prefix + "squeeze_excite/AvgPool"
    )
    conv1 = layers.Conv2D(
        Depth()(filters * se_ratio),
        kernel_size=1,
        padding="same",
        name=prefix + "squeeze_excite/Conv",
    )
    conv2 = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        name=prefix + "squeeze_excite/Conv_1",
    )
    relu = layers.ReLU(name=prefix + "squeeze_excite/Relu")
    hard_sigmoid = HardSigmoid()
    multiply = layers.Multiply(name=prefix + "squeeze_excite/Mul")

    def apply(inputs):
        x = ga_pool(inputs)
        x = conv1(x)
        x = relu(x)
        x = conv2(x)
        x = hard_sigmoid(x)
        x = multiply([inputs, x])
        return x

    return apply

def InvertedResBlock(
    expansion, filters, kernel_size, stride, se_ratio, activation, block_id, name=None
):
    if name is None:
        name = f"inverted_res_block_{backend.get_uid('inverted_res_block')}"

    def apply(x):
        shortcut = x
        prefix = "expanded_conv/"
        infilters = backend.int_shape(x)[channel_axis]
        if block_id:
            # Expand
            prefix = "expanded_conv_{}/".format(block_id)
            x = layers.Conv2D(
                Depth()(infilters * expansion),
                kernel_size=1,
                padding="same",
                use_bias=False,
                name=prefix + "expand",
            )(x)
            x = layers.BatchNormalization(
                axis=channel_axis,
                epsilon=1e-3,
                momentum=0.999,
                name=prefix + "expand/BatchNorm",
            )(x)
            x = activation(x)

        x = layers.DepthwiseConv2D(
            kernel_size,
            strides=stride,
            padding="same" if stride == 1 else "valid",
            use_bias=False,
            name=prefix + "depthwise",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + "depthwise/BatchNorm",
        )(x)
        x = activation(x)

        if se_ratio:
            x = SqueezeAndExcitationBlock(
                Depth()(infilters * expansion), se_ratio, prefix
            )(x)

        x = layers.Conv2D(
            filters,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name=prefix + "project",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + "project/BatchNorm",
        )(x)

        if stride == 1 and infilters == filters:
            x = layers.Add(name=prefix + "Add")([shortcut, x])
        return x

    return apply


def MobileNetV3(
    stack_fn,
    last_point_ch,
    input_shape=(None, None, 3),
    alpha=1.0,
    include_top=True,
    weights=None,
    classes=None,
    pooling=None,
    dropout_rate=0.2,
    classifier_activation="softmax",
    include_rescaling=True,
    minimalistic=True,
    name=None,
    **kwargs,
):

    if weights and not tf.io.gfile.exists(weights):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` or the path to the weights file to be loaded. "
            f"Weights file not found at location: {weights}"
        )

    if include_top and not classes:
        raise ValueError(
            "If `include_top` is True, "
            "you should specify `classes`. "
            f"Received: classes={classes}"
        )

    if minimalistic:
        kernel = 3
        activation = layers.ReLU()
        se_ratio = None
    else:
        kernel = 5
        activation = HardSwish()
        se_ratio = 0.25

    inputs = layers.Input(shape=input_shape)

    x = inputs

    if include_rescaling:
        x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(x)
    x = layers.Conv2D(
        16,
        kernel_size=3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        name="Conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name="Conv/BatchNorm"
    )(x)
    x = activation(x)

    x = stack_fn(x, kernel, activation, se_ratio)

    last_conv_ch = Depth()(backend.int_shape(x)[channel_axis] * 6)

    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_point_ch = Depth()(last_point_ch * alpha)
    x = layers.Conv2D(
        last_conv_ch,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name="Conv_1",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name="Conv_1/BatchNorm"
    )(x)
    x = activation(x)
    if include_top:
        x = layers.GlobalAveragePooling2D(keepdims=True)(x)
        x = layers.Conv2D(
            last_point_ch,
            kernel_size=1,
            padding="same",
            use_bias=True,
            name="Conv_2",
        )(x)
        x = activation(x)

        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv2D(classes, kernel_size=1, padding="same", name="Logits")(x)
        x = layers.Flatten()(x)
        x = layers.Activation(activation=classifier_activation, name="Predictions")(x)
    elif pooling == "avg":
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    elif pooling == "max":
        x = layers.GlobalMaxPooling2D(name="max_pool")(x)

    model = keras.Model(inputs, x, name=name, **kwargs)

    if weights is not None:
        model.load_weights(weights)
    return model


def MobileNetV3Small(
    input_shape=(None, None, 3),
    alpha=1.0,
    minimalistic=False,
    include_top=True,
    weights=None,
    classes=None,
    pooling=None,
    dropout_rate=0.2,
    classifier_activation="softmax",
    include_rescaling=True,
    name="MobileNetV3Small",
    **kwargs,
):
    def stack_fn(x, kernel, activation, se_ratio):
        def depth(d):
            return Depth()(d * alpha)

        x = InvertedResBlock(1, depth(16), 3, 2, se_ratio, layers.ReLU(), 0)(x)
        x = InvertedResBlock(72.0 / 16, depth(24), 3, 2, None, layers.ReLU(), 1)(x)
        x = InvertedResBlock(88.0 / 24, depth(24), 3, 1, None, layers.ReLU(), 2)(x)
        x = InvertedResBlock(4, depth(40), kernel, 2, se_ratio, activation, 3)(x)
        x = InvertedResBlock(6, depth(40), kernel, 1, se_ratio, activation, 4)(x)
        x = InvertedResBlock(6, depth(40), kernel, 1, se_ratio, activation, 5)(x)
        x = InvertedResBlock(3, depth(48), kernel, 1, se_ratio, activation, 6)(x)
        x = InvertedResBlock(3, depth(48), kernel, 1, se_ratio, activation, 7)(x)
        x = InvertedResBlock(6, depth(96), kernel, 2, se_ratio, activation, 8)(x)
        x = InvertedResBlock(6, depth(96), kernel, 1, se_ratio, activation, 9)(x)
        x = InvertedResBlock(6, depth(96), kernel, 1, se_ratio, activation, 10)(x)
        return x

    return MobileNetV3(
        stack_fn,
        1024,
        input_shape,
        alpha,
        include_top,
        weights,
        classes,
        pooling,
        dropout_rate,
        classifier_activation,
        include_rescaling,
        minimalistic,
        name=name,
        **kwargs,
    )


def MobileNetV3Large(
    input_shape=(None, None, 3),
    alpha=1.0,
    minimalistic=False,
    include_top=True,
    weights=None,
    classes=None,
    pooling=None,
    dropout_rate=0.2,
    classifier_activation="softmax",
    include_rescaling=True,
    name="MobileNetV3Large",
    **kwargs,
):
    def stack_fn(x, kernel, activation, se_ratio):
        def depth(d):
            return Depth()(d * alpha)

        x = InvertedResBlock(1, depth(16), 3, 1, None, layers.ReLU(), 0)(x)
        x = InvertedResBlock(4, depth(24), 3, 2, None, layers.ReLU(), 1)(x)
        x = InvertedResBlock(3, depth(24), 3, 1, None, layers.ReLU(), 2)(x)
        x = InvertedResBlock(3, depth(40), kernel, 2, se_ratio, layers.ReLU(), 3)(x)
        x = InvertedResBlock(3, depth(40), kernel, 1, se_ratio, layers.ReLU(), 4)(x)
        x = InvertedResBlock(3, depth(40), kernel, 1, se_ratio, layers.ReLU(), 5)(x)
        x = InvertedResBlock(6, depth(80), 3, 2, None, activation, 6)(x)
        x = InvertedResBlock(2.5, depth(80), 3, 1, None, activation, 7)(x)
        x = InvertedResBlock(2.3, depth(80), 3, 1, None, activation, 8)(x)
        x = InvertedResBlock(2.3, depth(80), 3, 1, None, activation, 9)(x)
        x = InvertedResBlock(6, depth(112), 3, 1, se_ratio, activation, 10)(x)
        x = InvertedResBlock(6, depth(112), 3, 1, se_ratio, activation, 11)(x)
        x = InvertedResBlock(6, depth(160), kernel, 2, se_ratio, activation, 12)(x)
        x = InvertedResBlock(6, depth(160), kernel, 1, se_ratio, activation, 13)(x)
        x = InvertedResBlock(6, depth(160), kernel, 1, se_ratio, activation, 14)(x)
        return x

    return MobileNetV3(
        stack_fn,
        1280,
        input_shape,
        alpha,
        include_top,
        weights,
        classes,
        pooling,
        dropout_rate,
        classifier_activation,
        include_rescaling,
        minimalistic,
        name=name,
        **kwargs,
    )
