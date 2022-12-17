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

from keras_cv.layers.group_conv2d import GroupConv2D
from keras_cv.models import utils

MODEL_CONFIGS = {
    # The naming pattern should be different
    "ResNeXt50_32x4d": {
        "num_blocks": [3, 4, 6, 3],
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


def ConvBlock(inputs, filters, kernel_size, strides, padding, name=None):
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

    def apply(inputs):
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            name=name + "_0_conv",
        )(inputs)
        x = tf.keras.layers.BatchNormalization(name=name + "bn_0")(x)
        x = tf.keras.layers.Activation('relu', name=name + "act_1")(x)
        return x

    return apply

# BottleNeck -> Bottleneck ?
def ResNeXt_Bottleneck(inputs, filters, strides, groups, bottleneck_width, name=None):
    # Use argument names for readability
    x = ConvBlock(
        inputs=inputs,
        filters=filters,
        kernel_size=1,
        strides=1,
        padding="same",
        name=name + "conv_block_1",
    )
    x = GroupConv2D(
        input_channels=filters,
        output_channels=filters,
        kernel_size=(3, 3),
        strides=strides,
        padding="same",
        groups=groups,
        bottleneck_width=bottleneck_width,
    )(x)
    x = tf.keras.layers.BatchNormalization(name=name + "bn_2")(x)
    x = tf.keras.layers.Activation('relu', name=name + "act_2")(x)
    x = ConvBlock(x, 2 * filters, (1, 1), 1, "same", name=name + "conv_block_2")
    shortcut = ConvBlock(
        inputs,
        filters=2 * filters,
        kernel_size=(1, 1),
        strides=strides,
        padding="same",
        name=name + "conv_block_3",
    )
    outputs = tf.keras.layers.add([x, shortcut])
    return outputs


def ResNeXt_Block(
    inputs, filters, strides, groups, num_blocks, bottleneck_width, name=None
):
    x = ResNeXt_Bottleneck(
        inputs=inputs,
        filters=filters,
        strides=strides,
        groups=groups,
        bottleneck_width=bottleneck_width,
        name='block_1',
    )
    for _ in range(1, num_blocks):
        x = ResNeXt_Bottleneck(
            x,
            filters=filters,
            strides=1,
            groups=groups,
            bottleneck_width=bottleneck_width,
            name=f'block_{_}',
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
    x = tf.keras.layers.Activation('relu', name=name + "post_relu")(x)
    x = tf.keras.layers.MaxPool2D(
        pool_size=(3, 3), strides=2, padding="same", name="max_pool"
    )(x)

    x = ResNeXt_Block(
        x,
        filters=128,
        strides=1,
        groups=cardinality,
        num_blocks=num_blocks[0],
        bottleneck_width=bottleneck_width,
        name=name,
    )
    x = ResNeXt_Block(
        x,
        filters=256,
        strides=2,
        groups=cardinality,
        num_blocks=num_blocks[1],
        bottleneck_width=bottleneck_width,
        name=name,
    )
    x = ResNeXt_Block(
        x,
        filters=512,
        strides=2,
        groups=cardinality,
        num_blocks=num_blocks[2],
        bottleneck_width=bottleneck_width,
        name=name,
    )
    x = ResNeXt_Block(
        x,
        filters=1024,
        strides=2,
        groups=cardinality,
        num_blocks=num_blocks[3],
        bottleneck_width=bottleneck_width,
        name=name,
    )

    # Unhardcoded the classifier activation
    # allowed for flexibility with top and a lack thereof
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
