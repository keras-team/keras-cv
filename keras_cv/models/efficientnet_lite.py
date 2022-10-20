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


"""EfficientNet Lite models for Keras.

Reference:
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
        https://arxiv.org/abs/1905.11946) (ICML 2019)
    - [Based on the original EfficientNet Lite's](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite)
"""
import copy
import math

import tensorflow as tf
from keras import backend
from keras import layers

from keras_cv.models import utils
from keras_cv.models.weights import parse_weights

DEFAULT_BLOCKS_ARGS = [
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 32,
        "filters_out": 16,
        "expand_ratio": 1,
        "id_skip": True,
        "strides": 1,
    },
    {
        "kernel_size": 3,
        "repeats": 2,
        "filters_in": 16,
        "filters_out": 24,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
    },
    {
        "kernel_size": 5,
        "repeats": 2,
        "filters_in": 24,
        "filters_out": 40,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
    },
    {
        "kernel_size": 3,
        "repeats": 3,
        "filters_in": 40,
        "filters_out": 80,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
    },
    {
        "kernel_size": 5,
        "repeats": 3,
        "filters_in": 80,
        "filters_out": 112,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
    },
    {
        "kernel_size": 5,
        "repeats": 4,
        "filters_in": 112,
        "filters_out": 192,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 192,
        "filters_out": 320,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
    },
]
CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 2.0,
        "mode": "fan_out",
        "distribution": "truncated_normal",
    },
}

DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 1.0 / 3.0,
        "mode": "fan_out",
        "distribution": "uniform",
    },
}

BASE_DOCSTRING = """Instantiates the {name} architecture.

    Reference:
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
        https://arxiv.org/abs/1905.11946) (ICML 2019)

    This function returns a Keras image classification model.

    For image classification use cases, see
    [this page for detailed examples](
    https://keras.io/api/applications/#usage-examples-for-image-classification-models).

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](
    https://keras.io/guides/transfer_learning/).

    Args:
        include_rescaling: whether or not to Rescale the inputs.If set to True,
                    inputs will be passed through a `Rescaling(1/255.0)` layer.
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: One of `None` (random initialization),
                or the path to the weights file to be loaded.
        input_shape: Optional shape tuple.
            It should have exactly 3 inputs channels.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`. Defaults to None.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified. Defaults to None.
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.
            Defaults to 'softmax'.
            When loading pretrained weights, `classifier_activation` can only
            be `None` or `"softmax"`.

    Returns:
        A `keras.Model` instance.
"""

BN_AXIS = 3


def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    Args:
      inputs: Input tensor.
      kernel_size: An integer or tuple/list of 2 integers.
    Returns:
      A tuple.
    """
    img_dim = 1
    input_size = backend.int_shape(inputs)[img_dim : (img_dim + 2)]
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


def round_filters(filters, depth_divisor, width_coefficient):
    """Round number of filters based on depth multiplier."""
    filters *= width_coefficient
    new_filters = max(
        depth_divisor,
        int(filters + depth_divisor / 2) // depth_divisor * depth_divisor,
    )
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))


def EfficientNetLiteBlock(
    activation="relu6",
    drop_rate=0.0,
    name="",
    filters_in=32,
    filters_out=16,
    kernel_size=3,
    strides=1,
    expand_ratio=1,
    id_skip=True,
):
    """An inverted residual block, without SE phase.

    Args:
        inputs: input tensor.
        activation: activation function.
        drop_rate: float between 0 and 1, fraction of the input units to drop.
        name: string, block label.
        filters_in: integer, the number of input filters.
        filters_out: integer, the number of output filters.
        kernel_size: integer, the dimension of the convolution window.
        strides: integer, the stride of the convolution.
        expand_ratio: integer, scaling coefficient for the input filters.
        id_skip: boolean.

    Returns:
        output tensor for the block.
    """

    def apply(inputs):
        # Expansion phase
        filters = filters_in * expand_ratio
        if expand_ratio != 1:
            x = layers.Conv2D(
                filters,
                1,
                padding="same",
                use_bias=False,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "expand_conv",
            )(inputs)
            x = layers.BatchNormalization(axis=BN_AXIS, name=name + "expand_bn")(x)
            x = layers.Activation(activation, name=name + "expand_activation")(x)
        else:
            x = inputs

        # Depthwise Convolution
        if strides == 2:
            x = layers.ZeroPadding2D(
                padding=correct_pad(x, kernel_size),
                name=name + "dwconv_pad",
            )(x)
            conv_pad = "valid"
        else:
            conv_pad = "same"
        x = layers.DepthwiseConv2D(
            kernel_size,
            strides=strides,
            padding=conv_pad,
            use_bias=False,
            depthwise_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "dwconv",
        )(x)
        x = layers.BatchNormalization(axis=BN_AXIS, name=name + "bn")(x)
        x = layers.Activation(activation, name=name + "activation")(x)

        # Skip SE block
        # Output phase
        x = layers.Conv2D(
            filters_out,
            1,
            padding="same",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "project_conv",
        )(x)
        x = layers.BatchNormalization(axis=BN_AXIS, name=name + "project_bn")(x)
        if id_skip and strides == 1 and filters_in == filters_out:
            if drop_rate > 0:
                x = layers.Dropout(
                    drop_rate, noise_shape=(None, 1, 1, 1), name=name + "drop"
                )(x)
            x = layers.add([x, inputs], name=name + "add")
        return x

    return apply


def EfficientNetLite(
    include_rescaling,
    include_top,
    width_coefficient,
    depth_coefficient,
    default_size,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    activation="relu6",
    blocks_args="default",
    model_name="efficientnetlite",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
    classifier_activation="softmax",
):
    """Instantiates the EfficientNetLite architecture using given scaling coefficients.

    Args:
        include_rescaling: whether to Rescale the inputs. If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        default_size: integer, default input image size.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: integer, a unit of network width.
        activation: activation function.
        blocks_args: list of dicts, parameters to construct block modules.
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

        Returns:
            A `keras.Model` instance.

        Raises:
            ValueError: in case of invalid argument for `weights`,
                or invalid input shape.
            ValueError: if `classifier_activation` is not `softmax` or `None` when
                using a pretrained top layer.
    """

    if blocks_args == "default":
        blocks_args = DEFAULT_BLOCKS_ARGS

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

    img_input = utils.parse_model_inputs(input_shape, input_tensor)

    # Build stem
    x = img_input

    if include_rescaling:
        # Use common rescaling strategy across keras_cv
        x = layers.Rescaling(1.0 / 255.0)(x)

    x = layers.ZeroPadding2D(padding=correct_pad(x, 3), name="stem_conv_pad")(x)
    x = layers.Conv2D(
        32,
        3,
        strides=2,
        padding="valid",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name="stem_conv",
    )(x)
    x = layers.BatchNormalization(axis=BN_AXIS, name="stem_bn")(x)
    x = layers.Activation(activation, name="stem_activation")(x)

    # Build blocks
    blocks_args = copy.deepcopy(blocks_args)

    b = 0
    blocks = float(sum(args["repeats"] for args in blocks_args))

    for (i, args) in enumerate(blocks_args):
        assert args["repeats"] > 0
        # Update block input and output filters based on depth multiplier.
        args["filters_in"] = round_filters(
            filters=args["filters_in"],
            width_coefficient=width_coefficient,
            depth_divisor=depth_divisor,
        )
        args["filters_out"] = round_filters(
            filters=args["filters_out"],
            width_coefficient=width_coefficient,
            depth_divisor=depth_divisor,
        )

        if i == 0 or i == (len(blocks_args) - 1):
            repeats = args.pop("repeats")
        else:
            repeats = round_repeats(
                repeats=args.pop("repeats"), depth_coefficient=depth_coefficient
            )

        for j in range(repeats):
            # The first block needs to take care of stride and filter size
            # increase.
            if j > 0:
                args["strides"] = 1
                args["filters_in"] = args["filters_out"]
            x = EfficientNetLiteBlock(
                activation=activation,
                drop_rate=drop_connect_rate * b / blocks,
                name="block{}{}_".format(i + 1, chr(j + 97)),
                **args,
            )(x)

            b += 1

    # Build top
    x = layers.Conv2D(
        1280,
        1,
        padding="same",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name="top_conv",
    )(x)
    x = layers.BatchNormalization(axis=BN_AXIS, name="top_bn")(x)
    x = layers.Activation(activation, name="top_activation")(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name="top_dropout")(x)
        x = layers.Dense(
            classes,
            activation=classifier_activation,
            kernel_initializer=DENSE_KERNEL_INITIALIZER,
            name="predictions",
        )(x)
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D(name="max_pool")(x)

    inputs = img_input

    # Create model.
    model = tf.keras.Model(inputs, x, name=model_name)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model


def EfficientNetLiteB0(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    **kwargs,
):
    return EfficientNetLite(
        include_rescaling,
        include_top,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=224,
        dropout_rate=0.2,
        model_name="efficientnetliteb0",
        weights=parse_weights(weights, include_top, "efficientnetliteb0"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def EfficientNetLiteB1(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    **kwargs,
):
    return EfficientNetLite(
        include_rescaling,
        include_top,
        width_coefficient=1.0,
        depth_coefficient=1.1,
        default_size=240,
        dropout_rate=0.2,
        model_name="efficientnetliteb1",
        weights=parse_weights(weights, include_top, "efficientnetliteb1"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def EfficientNetLiteB2(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    **kwargs,
):
    return EfficientNetLite(
        include_rescaling,
        include_top,
        width_coefficient=1.1,
        depth_coefficient=1.2,
        default_size=260,
        dropout_rate=0.3,
        model_name="efficientnetliteb2",
        weights=parse_weights(weights, include_top, "efficientnetliteb2"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def EfficientNetLiteB3(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    **kwargs,
):
    return EfficientNetLite(
        include_rescaling,
        include_top,
        width_coefficient=1.2,
        depth_coefficient=1.4,
        default_size=280,
        dropout_rate=0.3,
        model_name="efficientnetliteb3",
        weights=parse_weights(weights, include_top, "efficientnetliteb3"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def EfficientNetLiteB4(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    **kwargs,
):
    return EfficientNetLite(
        include_rescaling,
        include_top,
        width_coefficient=1.4,
        depth_coefficient=1.8,
        default_size=300,
        dropout_rate=0.3,
        model_name="efficientnetliteb4",
        weights=parse_weights(weights, include_top, "efficientnetliteb4"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


EfficientNetLiteB0.__doc__ = BASE_DOCSTRING.format(name="EfficientNetLiteB0")
EfficientNetLiteB1.__doc__ = BASE_DOCSTRING.format(name="EfficientNetLiteB1")
EfficientNetLiteB2.__doc__ = BASE_DOCSTRING.format(name="EfficientNetLiteB2")
EfficientNetLiteB3.__doc__ = BASE_DOCSTRING.format(name="EfficientNetLiteB3")
EfficientNetLiteB4.__doc__ = BASE_DOCSTRING.format(name="EfficientNetLiteB4")
