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

import math

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers

from keras_cv.models import utils

MODEL_CONFIGS = {
    # The naming pattern should be different
    "ResNeXt50_32x4d": {
        "num_blocks": [3, 4, 6, 3],
        "stackwise_filters": [128, 256, 512, 1024],
        "stackwise_strides": [1, 2, 2, 2],
        "cardinality": 32,
        "bottleneck_width": 4,
    },
    "ResNeXt50_64x4d": {
        "num_blocks": [3, 4, 6, 3],
        "cardinality": 64,
        "bottleneck_width": 4,
    },
    "ResNeXt101_32x4d": {
        "num_blocks": [3, 4, 23, 3],
        "cardinality": 32,
        "bottleneck_width": 4,
    },
    "ResNeXt101_32x8d": {
        "num_blocks": [3, 4, 23, 3],
        "cardinality": 32,
        "bottleneck_width": 8,
    },
    "ResNeXt101_64x4d": {
        "num_blocks": [3, 4, 23, 3],
        "cardinality": 64,
        "bottleneck_width": 4,
    },
}


def ConvBlock(filters, kernel_size, strides, padding, name=None):
    """A basic block.
    Args:
        filters: integer, filters of the basic layer.
        kernel_size: kernel size of the bottleneck layer.
        stride: stride of the first layer.
        padding:
        name: string, block label.
    Returns:
      Output tensor for the conv block.
    """

    def apply(x):
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            name=name + "_0_conv",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=name + "bn_0")(x)
        x = tf.keras.layers.Activation("relu", name=name + "act_1")(x)
        return x

    return apply


# BottleNeck -> Bottleneck ?
def ResNeXt_Bottleneck(inputs, filters, strides, groups, bottleneck_width, name=None):
    # Use argument names for readability
    x = ConvBlock(
        filters=filters,
        kernel_size=1,
        strides=1,
        padding="same",
        name=name + "conv_block_1",
    )(inputs)
    D = math.floor((filters / 4) * (bottleneck_width / 64))
    x = layers.Conv2D(
        filters=groups * D,
        kernel_size=(3, 3),
        strides=strides,
        padding="same",
        groups=groups,
    )(x)
    x = tf.keras.layers.BatchNormalization(name=name + "bn_2")(x)
    x = tf.keras.layers.Activation("relu", name=name + "act_2")(x)
    x = ConvBlock(
        filters=2 * filters,
        kernel_size=1,
        strides=1,
        padding="same",
        name=name + "conv_block_2",
    )(x)
    shortcut = ConvBlock(
        filters=2 * filters,
        kernel_size=(1, 1),
        strides=strides,
        padding="same",
        name=name + "conv_block_3",
    )(inputs)
    outputs = tf.keras.layers.add([x, shortcut])
    return outputs


def ResNeXt_Block(
    inputs, filters, strides, groups, num_blocks, bottleneck_width, name=None
):
    x = inputs
    for _ in range(0, num_blocks):
        x = ResNeXt_Bottleneck(
            x,
            filters=filters,
            strides=strides if _ == 0 else 1,
            groups=groups,
            bottleneck_width=bottleneck_width,
            name=name + f"bottleneck_block_{_}",
        )
    return x


def ResNeXt(
    include_rescaling,
    include_top,
    name="ResNeXt",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
    classifier_activation="softmax",
    block_fn=None,
    num_blocks=None,
    cardinality=None,
    stackwise_filters=None,
    stackwise_strides=None,
    bottleneck_width=None,
):
    if weights and not tf.io.gfile.exists(weights):
        raise ValueError(
            "The `weights` argument should be either `None` or the path to the "
            "weights file to be loaded. Weights file not found at location: {weights}"
        )

    if include_top and not classes:
        raise ValueError(
            "If `include_top` is True, you should specify `classes`. "
            f"Received: classes={classes}"
        )

    if include_top and pooling:
        raise ValueError(
            f"`pooling` must be `None` when `include_top=True`."
            f"Received pooling={pooling} and include_top={include_top}. "
        )

    inputs = utils.parse_model_inputs(input_shape, input_tensor)
    x = inputs

    if include_rescaling:
        x = layers.Rescaling(1 / 255.0)(x)

    x = tf.keras.layers.Conv2D(
        filters=64, kernel_size=(7, 7), strides=2, padding="same", name="post_conv2d"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="post_bn")(x)
    x = tf.keras.layers.Activation("relu", name=name + "post_relu")(x)
    x = tf.keras.layers.MaxPool2D(
        pool_size=(3, 3), strides=2, padding="same", name="max_pool"
    )(x)

    num_stacks = len(stackwise_filters)
    for stack_index in range(num_stacks):
        x = ResNeXt_Block(
            inputs=x,
            groups=cardinality,
            filters=stackwise_filters[stack_index],
            num_blocks=num_blocks[stack_index],
            strides=stackwise_strides[stack_index],
            bottleneck_width=bottleneck_width,
            name=f"resnext_block_{stack_index}",
        )

    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = layers.Dense(classes, activation=classifier_activation, name="predictions")(
            x
        )
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D(name="max_pool")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


def ResNeXt50_32x4d(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="ResNeXt50_32x4d",
    **kwargs,
):

    """Instantiates the ResNext50_32 architecture."""
    return ResNeXt(
        num_blocks=MODEL_CONFIGS["ResNeXt50_32x4d"]["num_blocks"],
        cardinality=MODEL_CONFIGS["ResNeXt50_32x4d"]["cardinality"],
        bottleneck_width=MODEL_CONFIGS["ResNeXt50_32x4d"]["bottleneck_width"],
        stackwise_filters=MODEL_CONFIGS["ResNeXt50_32x4d"]["stackwise_filters"],
        stackwise_strides=MODEL_CONFIGS["ResNeXt50_32x4d"]["stackwise_strides"],
        name=name,
        include_rescaling=include_rescaling,
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ResNeXt50_64x4d(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="ResNeXt50_64x4d",
    **kwargs,
):

    """Instantiates the ResNeXt50_64x4d architecture."""
    return ResNeXt(
        num_blocks=MODEL_CONFIGS["ResNeXt50_64x4d"]["num_blocks"],
        cardinality=MODEL_CONFIGS["ResNeXt50_64x4d"]["cardinality"],
        bottleneck_width=MODEL_CONFIGS["ResNeXt50_64x4d"]["bottleneck_width"],
        name=name,
        include_rescaling=include_rescaling,
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ResNeXt101_32x4d(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="ResNeXt101_32x4d",
    **kwargs,
):

    """Instantiates the ResNeXt101_32x4d architecture."""
    return ResNeXt(
        num_blocks=MODEL_CONFIGS["ResNeXt101_32x4d"]["num_blocks"],
        cardinality=MODEL_CONFIGS["ResNeXt101_32x4d"]["cardinality"],
        bottleneck_width=MODEL_CONFIGS["ResNeXt101_32x4d"]["bottleneck_width"],
        name=name,
        include_rescaling=include_rescaling,
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ResNeXt101_32x8d(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="ResNeXt101_32x8d",
    **kwargs,
):

    """Instantiates the ResNeXt101_32x8d architecture."""
    return ResNeXt(
        num_blocks=MODEL_CONFIGS["ResNeXt101_32x8d"]["num_blocks"],
        cardinality=MODEL_CONFIGS["ResNeXt101_32x8d"]["cardinality"],
        bottleneck_width=MODEL_CONFIGS["ResNeXt101_32x8d"]["bottleneck_width"],
        name=name,
        include_rescaling=include_rescaling,
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ResNeXt101_64x4d(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="ResNeXt101_64x4d",
    **kwargs,
):

    """Instantiates the ResNeXt101_64x4d architecture."""
    return ResNeXt(
        num_blocks=MODEL_CONFIGS["ResNeXt101_64x4d"]["num_blocks"],
        cardinality=MODEL_CONFIGS["ResNeXt101_64x4d"]["cardinality"],
        bottleneck_width=MODEL_CONFIGS["ResNeXt101_64x4d"]["bottleneck_width"],
        name=name,
        include_rescaling=include_rescaling,
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
