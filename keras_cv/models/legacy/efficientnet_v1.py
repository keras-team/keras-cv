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


"""EfficientNet models for Keras.

Reference:
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
        (ICML 2019)
    - [Based on the original keras.applications EfficientNet](https://github.com/keras-team/keras/blob/master/keras/applications/efficientnet.py)
"""  # noqa: E501

import copy
import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers

from keras_cv.models.legacy import utils
from keras_cv.models.legacy.weights import parse_weights

DEFAULT_BLOCKS_ARGS = [
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 32,
        "filters_out": 16,
        "expand_ratio": 1,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 2,
        "filters_in": 16,
        "filters_out": 24,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 2,
        "filters_in": 24,
        "filters_out": 40,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 3,
        "filters_in": 40,
        "filters_out": 80,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 3,
        "filters_in": 80,
        "filters_out": 112,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 4,
        "filters_in": 112,
        "filters_out": 192,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 192,
        "filters_out": 320,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
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
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
        (ICML 2019)

    This class represents a Keras image classification model.

    For image classification use cases, see
    [this page for detailed examples](https://keras.io/api/applications/#usage-examples-for-image-classification-models).

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        include_rescaling: bool, whether to rescale the inputs. If set to
            True, inputs will be passed through a `Rescaling(1/255.0)` layer.
        include_top: bool, Whether to include the fully-connected layer at the
            top of the network.
        weights: One of `None` (random initialization), or the path to the
            weights file to be loaded.
        input_shape: tuple, Optional shape tuple. It should have exactly 3
            inputs channels.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to
            use as image input for the model.
        pooling: Optional pooling mode for feature extraction when `include_top`
            is `False`, defaults to None.
            - `None` means that the output of the model will be the 4D tensor
                output of the last convolutional layer.
            - `avg` means that global average pooling will be applied to the
                output of the last convolutional layer, and thus the output of
                the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        num_classes: int, Optional number of classes to classify images into,
            only to be specified if `include_top` is True, and if no `weights`
            argument is specified, defaults to None.
        classifier_activation: A `str` or callable. The activation function to
            use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top"
            layer. Defaults to 'softmax'. When loading pretrained weights,
            `classifier_activation` can only be `None` or `"softmax"`.

    Returns:
        A `keras.Model` instance.
"""  # noqa: E501

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


def apply_conv_bn(
    x,
    conv_type,
    filters,
    kernel_size,
    strides=1,
    padding="same",
    use_bias=False,
    kernel_initializer=CONV_KERNEL_INITIALIZER,
    bn_norm=True,
    activation="swish",
    name="",
):
    """
    Represents Convolutional Block with optional Batch Normalization layer and
    activation layer

    Args:
        x: Tensor
        conv_type: str, Type of Conv layer to be used in block.
            - 'normal': The Conv2D layer will be used.
            - 'depth': The DepthWiseConv2D layer will be used.
        filters: int, The filter size of the Conv layer. It should be `None`
            when `conv_type` is set as `depth`
        kernel_size: int (or) tuple, The kernel size of the Conv layer.
        strides: int (or) tuple, The stride value of Conv layer.
        padding: str (or) callable, The type of padding for Conv layer.
        use_bias: bool, Boolean to use bias for Conv layer.
        kernel_initializer: dict (or) str (or) callable, The kernel initializer
            for Conv layer.
        bn_norm: bool, Boolean to add BatchNormalization layer after Conv layer.
        activation: str (or) callable, Activation to be applied on the output at
            the end.
        name: str, name of the block

    Returns:
        tf.Tensor
    """
    if conv_type == "normal":
        if filters is None or kernel_size is None:
            raise ValueError(
                "The filter size and kernel size should be set for Conv2D "
                "layer."
            )
        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            name=name + "_conv",
        )(x)
    elif conv_type == "depth":
        if filters is not None:
            raise ValueError(
                "Filter size shouldn't be set for DepthWiseConv2D layer."
            )
        if kernel_size is None or strides is None:
            raise ValueError(
                "The kernel size and strides should be set for DepthWiseConv2D "
                "layer."
            )
        x = layers.DepthwiseConv2D(
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            depthwise_initializer=kernel_initializer,
            name=name + "_dwconv",
        )(x)
    else:
        raise ValueError(
            "The 'conv_type' parameter should be set either to 'normal' or "
            "'depth'"
        )

    if bn_norm:
        x = layers.BatchNormalization(axis=BN_AXIS, name=name + "_bn")(x)
    if activation is not None:
        x = layers.Activation(activation, name=name + "_activation")(x)

    return x


def apply_efficientnet_block(
    inputs,
    filters_in=32,
    filters_out=16,
    kernel_size=3,
    strides=1,
    activation="swish",
    expand_ratio=1,
    se_ratio=0.0,
    id_skip=True,
    drop_rate=0.0,
    name="",
):
    """An inverted residual block.

    Args:
        inputs: Tensor, The input tensor of the block
        filters_in: integer, the number of input filters.
        filters_out: integer, the number of output filters.
        kernel_size: integer, the dimension of the convolution window.
        strides: integer, the stride of the convolution.
        activation: activation function.
        expand_ratio: integer, scaling coefficient for the input filters.
        se_ratio: float between 0 and 1, fraction to squeeze the input filters.
        id_skip: boolean.
        drop_rate: float between 0 and 1, fraction of the input units to drop.
        name: string, block label.

    Returns:
        tf.Tensor
    """
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = apply_conv_bn(
            x=inputs,
            conv_type="normal",
            filters=filters,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            bn_norm=True,
            activation=activation,
            name=name + "_expand",
        )
    else:
        x = inputs

    # Depthwise Convolution
    if strides == 2:
        x = layers.ZeroPadding2D(
            padding=correct_pad(x, kernel_size),
            name=name + "_dwconv_pad",
        )(x)
        conv_pad = "valid"
    else:
        conv_pad = "same"

    x = apply_conv_bn(
        x=x,
        conv_type="depth",
        filters=None,
        kernel_size=kernel_size,
        strides=strides,
        padding=conv_pad,
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        bn_norm=True,
        activation=activation,
        name=name,
    )

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D(name=name + "_se_squeeze")(x)
        if BN_AXIS == 1:
            se_shape = (filters, 1, 1)
        else:
            se_shape = (1, 1, filters)
        se = layers.Reshape(se_shape, name=name + "_se_reshape")(se)
        se = layers.Conv2D(
            filters_se,
            1,
            padding="same",
            activation=activation,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "_se_reduce",
        )(se)
        se = layers.Conv2D(
            filters,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name + "_se_expand",
        )(se)
        x = layers.multiply([x, se], name=name + "_se_excite")

    # Output phase
    x = apply_conv_bn(
        x=x,
        conv_type="normal",
        filters=filters_out,
        kernel_size=1,
        padding="same",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        bn_norm=True,
        activation=None,
        name=name + "_project",
    )

    if id_skip and strides == 1 and filters_in == filters_out:
        if drop_rate > 0:
            x = layers.Dropout(
                drop_rate,
                noise_shape=(None, 1, 1, 1),
                name=name + "_drop",
            )(x)
        x = layers.add([x, inputs], name=name + "_add")

    return x


@keras.utils.register_keras_serializable(package="keras_cv.models")
class EfficientNet(keras.Model):
    """This class represents a Keras EfficientNet architecture.
    Args:
        include_rescaling: bool, whether to rescale the inputs. If set to
            True, inputs will be passed through a `Rescaling(1/255.0)` layer.
        include_top: bool, whether to include the fully-connected layer at the
            top of the network.
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        default_size: integer, default input image size.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: integer, a unit of network width.
        activation: activation function.
        blocks_args: list of dicts, parameters to construct block modules.
        model_name: string, model name.
        weights: one of `None` (random initialization), or the path to the
            weights file to be loaded.
        input_shape: optional shape tuple, it should have exactly 3 input
            channels.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to
            use as image input for the model.
        pooling: optional pooling mode for feature extraction when `include_top`
            is `False`.
            - `None` means that the output of the model will be the 4D tensor
                output of the last convolutional layer.
            - `avg` means that global average pooling will be applied to the
                output of the last convolutional layer, and thus the output of
                the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        num_classes: optional number of classes to classify images into,
            only to be specified if `include_top` is True, and if no `weights`
            argument is specified.
        classifier_activation: A `str` or callable. The activation function to
            use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top"
            layer.
    Returns:
      A `keras.Model` instance.
    Raises:
      ValueError: in case of invalid argument for `weights`, or invalid input
        shape.
      ValueError: if `classifier_activation` is not `softmax` or `None` when
        using a pretrained top layer.
    """

    def __init__(
        self,
        include_rescaling,
        include_top,
        width_coefficient,
        depth_coefficient,
        default_size,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
        depth_divisor=8,
        activation="swish",
        blocks_args="default",
        model_name="efficientnet",
        weights=None,
        input_shape=(None, None, 3),
        input_tensor=None,
        pooling=None,
        num_classes=None,
        classifier_activation="softmax",
        **kwargs,
    ):
        blocks_args_type = blocks_args

        if blocks_args == "default":
            blocks_args = DEFAULT_BLOCKS_ARGS

        if weights and not tf.io.gfile.exists(weights):
            raise ValueError(
                "The `weights` argument should be either `None` or the path to "
                "the weights file to be loaded. Weights file not found at "
                f"location: {weights}"
            )

        if include_top and not num_classes:
            raise ValueError(
                "If `include_top` is True, you should specify `num_classes`. "
                f"Received: num_classes={num_classes}"
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

        x = layers.ZeroPadding2D(
            padding=correct_pad(x, 3), name="stem_conv_pad"
        )(x)

        x = apply_conv_bn(
            x=x,
            conv_type="normal",
            filters=EfficientNet.round_filters(
                32, width_coefficient, depth_divisor
            ),
            kernel_size=3,
            strides=2,
            padding="valid",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            bn_norm=True,
            activation=activation,
            name="stem",
        )

        # Build blocks
        blocks_args = copy.deepcopy(blocks_args)

        b = 0
        blocks = float(
            sum(
                EfficientNet.round_repeats(args["repeats"], depth_coefficient)
                for args in blocks_args
            )
        )
        for i, args in enumerate(blocks_args):
            assert args["repeats"] > 0
            # Update block input and output filters based on depth multiplier.
            args["filters_in"] = EfficientNet.round_filters(
                args["filters_in"], width_coefficient, depth_divisor
            )
            args["filters_out"] = EfficientNet.round_filters(
                args["filters_out"], width_coefficient, depth_divisor
            )

            for j in range(
                EfficientNet.round_repeats(
                    args.pop("repeats"), depth_coefficient
                )
            ):
                # The first block needs to take care of stride and filter size
                # increase.
                if j > 0:
                    args["strides"] = 1
                    args["filters_in"] = args["filters_out"]
                x = apply_efficientnet_block(
                    inputs=x,
                    activation=activation,
                    drop_rate=drop_connect_rate * b / blocks,
                    name="block{}{}".format(i + 1, chr(j + 97)),
                    **args,
                )
                b += 1

        # Build top
        x = apply_conv_bn(
            x=x,
            conv_type="normal",
            filters=self.round_filters(1280, width_coefficient, depth_divisor),
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            bn_norm=True,
            activation=activation,
            name="top",
        )

        if include_top:
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate, name="top_dropout")(x)
            x = layers.Dense(
                num_classes,
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
        super().__init__(inputs=inputs, outputs=x, name=model_name, **kwargs)

        # Load weights.
        if weights is not None:
            self.load_weights(weights)

        self.include_rescaling = include_rescaling
        self.include_top = include_top
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.default_size = default_size
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
        self.depth_divisor = depth_divisor
        self.activation = activation
        self.blocks_args = blocks_args_type
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    @staticmethod
    def round_filters(filters, width_coefficient, divisor):
        """Round number of filters based on depth multiplier.
        Args:
            filters: int, number of filters for Conv layer
            width_coefficient: float, denotes the scaling coefficient of network
                width
            divisor: int, a unit of network width

        Returns:
            int, new rounded filters value for Conv layer
        """
        filters *= width_coefficient
        new_filters = max(
            divisor, int(filters + divisor / 2) // divisor * divisor
        )
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    @staticmethod
    def round_repeats(repeats, depth_coefficient):
        """Round number of repeats based on depth multiplier.
        Args:
            repeats: int, number of repeats of efficientnet block
            depth_coefficient: float, denotes the scaling coefficient of network
                depth

        Returns:
            int, rounded repeats
        """
        return int(math.ceil(depth_coefficient * repeats))

    def get_config(self):
        return {
            "include_rescaling": self.include_rescaling,
            "include_top": self.include_top,
            "width_coefficient": self.width_coefficient,
            "depth_coefficient": self.depth_coefficient,
            "default_size": self.default_size,
            "dropout_rate": self.dropout_rate,
            "drop_connect_rate": self.drop_connect_rate,
            "depth_divisor": self.depth_divisor,
            "activation": self.activation,
            "blocks_args": self.blocks_args,
            "input_tensor": self.input_tensor,
            "input_shape": self.input_shape[1:],
            "model_name": self.name,
            "pooling": self.pooling,
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def EfficientNetB0(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="efficientnetb0",
    **kwargs,
):
    return EfficientNet(
        include_rescaling,
        include_top,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=224,
        dropout_rate=0.2,
        model_name=name,
        weights=parse_weights(weights, include_top, "efficientnetb0"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def EfficientNetB1(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="efficientnetb1",
    **kwargs,
):
    return EfficientNet(
        include_rescaling,
        include_top,
        width_coefficient=1.0,
        depth_coefficient=1.1,
        default_size=240,
        dropout_rate=0.2,
        model_name=name,
        weights=parse_weights(weights, include_top, "efficientnetb1"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def EfficientNetB2(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="efficientnetb2",
    **kwargs,
):
    return EfficientNet(
        include_rescaling,
        include_top,
        width_coefficient=1.1,
        depth_coefficient=1.2,
        default_size=260,
        dropout_rate=0.3,
        model_name=name,
        weights=parse_weights(weights, include_top, "efficientnetb2"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def EfficientNetB3(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="efficientnetb3",
    **kwargs,
):
    return EfficientNet(
        include_rescaling,
        include_top,
        width_coefficient=1.2,
        depth_coefficient=1.4,
        default_size=300,
        dropout_rate=0.3,
        model_name=name,
        weights=parse_weights(weights, include_top, "efficientnetb3"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def EfficientNetB4(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="efficientnetb4",
    **kwargs,
):
    return EfficientNet(
        include_rescaling,
        include_top,
        width_coefficient=1.4,
        depth_coefficient=1.8,
        default_size=380,
        dropout_rate=0.4,
        model_name=name,
        weights=parse_weights(weights, include_top, "efficientnetb4"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def EfficientNetB5(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="efficientnetb5",
    **kwargs,
):
    return EfficientNet(
        include_rescaling,
        include_top,
        width_coefficient=1.6,
        depth_coefficient=2.2,
        default_size=456,
        dropout_rate=0.4,
        model_name=name,
        weights=parse_weights(weights, include_top, "efficientnetb5"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def EfficientNetB6(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="efficientnetb6",
    **kwargs,
):
    return EfficientNet(
        include_rescaling,
        include_top,
        width_coefficient=1.8,
        depth_coefficient=2.6,
        default_size=528,
        dropout_rate=0.5,
        model_name=name,
        weights=parse_weights(weights, include_top, "efficientnetb6"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def EfficientNetB7(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="efficientnetb7",
    **kwargs,
):
    return EfficientNet(
        include_rescaling,
        include_top,
        width_coefficient=2.0,
        depth_coefficient=3.1,
        default_size=600,
        dropout_rate=0.5,
        model_name=name,
        weights=parse_weights(weights, include_top, "efficientnetb7"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


EfficientNetB0.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB0")
EfficientNetB1.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB1")
EfficientNetB2.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB2")
EfficientNetB3.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB3")
EfficientNetB4.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB4")
EfficientNetB5.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB5")
EfficientNetB6.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB6")
EfficientNetB7.__doc__ = BASE_DOCSTRING.format(name="EfficientNetB7")
