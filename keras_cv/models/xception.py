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


"""Xception model for KerasCV.


"""

import copy
import math
import os

import tensorflow as tf
from keras import backend
from keras import layers

from keras_cv.models import utils
from keras_cv.models.weights import parse_weights

MODEL_CONFIG = {
    'Xception': {
        'entry_flow_filters': [32, 64, 128, 256, 728],
        'middle_flow_filters': [728 for _ in range(8)],
        'exit_flow_filters': [(728, 1024), 1536, 2048]
    }
}

CHANNEL_AXIS = -1


def Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding="same", separable=True, name=None):
    if name is None:
        name = f"conv_{backend.get_uid('conv')}"

    conv = layers.SeparableConv2D if separable else layers.Conv2D

    def apply(x):
        x = conv(filters, kernel_size, strides, padding=padding, use_bias=False, name=name)(x)
        x = layers.BatchNormalization(
            axis=CHANNEL_AXIS, name=f"{name}_bn"
        )(x)
        return x

    return apply


def Block(filters, kernel_size=(3, 3), strides=(1, 1), padding="same",
          activation="relu", activation_loc="first", separable=True, name=None):
    if name is None:
        name = f"block_{backend.get_uid('block')}"

    def apply(x):
        if activation_loc == "first":
            x = layers.Activation(activation, name=f"{name}_act")(x)
        x = Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            separable=separable,
            name=f"{name}",
        )(x)
        if activation_loc == "last":
            x = layers.Activation(activation, name=f"{name}_act")(x)
        return x

    return apply


def ResidualBlock(filters, kernel_size=(3, 3), num_blocks: int = 3, name=None):
    if name is None:
        name = f"residual_block_{backend.get_uid('residual_block_')}"

    def apply(x):
        residual = x
        for i in range(num_blocks):
            x = Block(filters, kernel_size, activation_loc="first", name=f"{name}_sepconv{i}")(x)
        x = layers.add([x, residual])
        return x

    return apply


def XceptionStack(filters, skip_first_act=False, name=None):
    if name is None:
        name = f"stack_{backend.get_uid('stack')}"

    if isinstance(filters, int):
        filters0 = filters
        filters1 = filters
    else:
        filters0, filters1 = filters

    activation_loc = "none" if skip_first_act else "first"

    def apply(x):
        residual = Conv2D(filters1, (1, 1), strides=(2, 2), padding="same", separable=False)(x)
        x = Block(filters0, activation_loc=activation_loc, name=f"{name}_sepconv1")(x)
        x = Block(filters1, name=f"{name}_sepconv2")(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name=f"{name}_pool")(x)
        x = layers.add([x, residual])
        return x

    return apply


def GeneralXception(
        entry_flow_filters,
        middle_flow_filters,
        exit_flow_filters,
        include_rescaling,
        include_top,
        model_name="xception",
        weights=None,
        input_shape=(None, None, 3),
        input_tensor=None,
        pooling=None,
        classes=None,
        classifier_activation="softmax",
        **kwargs,
):
    """Instantiates the Xception architecture using given scaling
        coefficients.

    Args:
        include_rescaling: whether or not to Rescale the inputs.If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        model_name: string, model name.
        weights: one of `None` (random initialization),
            or the path to the weights file to be loaded.
        input_shape: optional shape tuple,
            It should have exactly 3 inputs channels.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.
        entry_filters: TODO

    Returns:
      A `keras.Model` instance.

    Raises:
      ValueError: in case of invalid argument for `weights`,
        or invalid input shape.
      ValueError: if `classifier_activation` is not `"softmax"` or `None` when
        using a pretrained top layer.
    """

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

    # Determine proper input shape
    img_input = utils.parse_model_inputs(input_shape, input_tensor)

    x = img_input

    if include_rescaling:
        x = layers.Rescaling(scale=1 / 255.0)(x)

    # entry flow
    x = Block(entry_flow_filters[0], strides=(2, 2), padding="valid", activation_loc="last", separable=False, name="block1_conv1")(x)
    x = Block(entry_flow_filters[1], padding="valid", activation_loc="last", separable=False, name="block1_conv2")(x)

    for i, filters in enumerate(entry_flow_filters[2:]):
        x = XceptionStack(filters, skip_first_act=i == 0, name=f"entry_block_{i}")(x)

    # middle flow
    for i, filters in enumerate(middle_flow_filters):
        x = ResidualBlock(filters, name=f"middle_block_{i}")(x)

    for i, filters in enumerate(exit_flow_filters[:-2]):
        x = XceptionStack(filters, name=f"exit_block_{i}")(x)

    x = Block(exit_flow_filters[-2], (3, 3), activation_loc="last", name="block14_sepconv1")(x)
    x = Block(exit_flow_filters[-1], (3, 3), activation_loc="last", name="block14_sepconv2")(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = layers.Dense(
            classes, activation=classifier_activation, name="predictions"
        )(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D(name="max_pool")(x)

    inputs = img_input

    model = tf.keras.Model(inputs, x, name=model_name, **kwargs)

    if weights is not None:
        model.load_weights(weights)

    return model


def Xception(
        include_rescaling,
        include_top,
        model_name="xception",
        weights=None,
        input_shape=(None, None, 3),
        input_tensor=None,
        pooling=None,
        classes=None,
        classifier_activation="softmax",
        **kwargs,
):
    return GeneralXception(
        MODEL_CONFIG['Xception']['entry_flow_filters'],
        MODEL_CONFIG['Xception']['middle_flow_filters'],
        MODEL_CONFIG['Xception']['exit_flow_filters'],
        include_rescaling,
        include_top,
        model_name=model_name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )
