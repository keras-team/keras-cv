# Copyright 2023 The KerasCV Authors
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

"""EfficientNet V2 models for KerasCV.

Reference:
    - [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) (ICML 2021)
    - [Based on the original keras.applications EfficientNetV2](https://github.com/keras-team/keras/blob/master/keras/applications/efficientnet_v2.py)
"""  # noqa: E501

import copy
import math

from tensorflow import keras
from tensorflow.keras import layers

from keras_cv.layers import FusedMBConvBlock
from keras_cv.layers import MBConvBlock
from keras_cv.models import utils

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

BN_AXIS = 3

BASE_DOCSTRING = """Instantiates the {name} architecture.

    Reference:
    - [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
      (ICML 2021)

    This function returns a Keras image classification model.

    For image classification use cases, see
    [this page for detailed examples](https://keras.io/api/applications/#usage-examples-for-image-classification-models).

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        include_rescaling: bool, whether to rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
    Returns:
      A `keras.Model` instance.
"""  # noqa: E501


def round_filters(filters, width_coefficient, min_depth, depth_divisor):
    """Round number of filters based on depth multiplier."""
    filters *= width_coefficient
    minimum_depth = min_depth or depth_divisor
    new_filters = max(
        minimum_depth,
        int(filters + depth_divisor / 2) // depth_divisor * depth_divisor,
    )
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))


@keras.utils.register_keras_serializable(package="keras_cv.models")
class EfficientNetV2(keras.Model):
    """Instantiates the EfficientNetV2 architecture using given scaling
    coefficients.
    Args:
        include_rescaling: bool, whether to rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        default_size: integer, default input image size.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: integer, a unit of network width.
        min_depth: integer, minimum number of filters.
        bn_momentum: float. Momentum parameter for Batch Normalization layers.
        activation: activation function.
        blocks_args: list of dicts, parameters to construct block modules.
        model_name: string, model name.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
    """

    def __init__(
        self,
        *include_rescaling,
        width_coefficient,
        depth_coefficient,
        default_size,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
        depth_divisor=8,
        min_depth=8,
        bn_momentum=0.9,
        activation="swish",
        blocks_args="default",
        model_name="efficientnet",
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        if blocks_args == "default":
            blocks_args = DEFAULT_BLOCKS_ARGS[model_name]

        input_blocks_args = copy.deepcopy(blocks_args)

        # Determine proper input shape
        img_input = utils.parse_model_inputs(input_shape, input_tensor)

        x = img_input

        if include_rescaling:
            x = layers.Rescaling(scale=1 / 255.0)(x)

        # Build stem
        stem_filters = round_filters(
            filters=blocks_args[0]["input_filters"],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor,
        )
        x = layers.Conv2D(
            filters=stem_filters,
            kernel_size=3,
            strides=2,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            use_bias=False,
            name="stem_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=BN_AXIS,
            momentum=bn_momentum,
            name="stem_bn",
        )(x)
        x = layers.Activation(activation, name="stem_activation")(x)

        # Build blocks
        blocks_args = copy.deepcopy(blocks_args)
        b = 0
        blocks = float(sum(args["num_repeat"] for args in blocks_args))

        for i, args in enumerate(blocks_args):
            assert args["num_repeat"] > 0

            # Update block input and output filters based on depth multiplier.
            args["input_filters"] = round_filters(
                filters=args["input_filters"],
                width_coefficient=width_coefficient,
                min_depth=min_depth,
                depth_divisor=depth_divisor,
            )
            args["output_filters"] = round_filters(
                filters=args["output_filters"],
                width_coefficient=width_coefficient,
                min_depth=min_depth,
                depth_divisor=depth_divisor,
            )

            repeats = round_repeats(
                repeats=args.pop("num_repeat"),
                depth_coefficient=depth_coefficient,
            )
            for j in range(repeats):
                # The first block needs to take care of stride and filter size
                # increase.
                if j > 0:
                    args["strides"] = 1
                    args["input_filters"] = args["output_filters"]

                # Determine which conv type to use:
                block = {
                    0: MBConvBlock(
                        input_filters=args["input_filters"],
                        output_filters=args["output_filters"],
                        expand_ratio=args["expand_ratio"],
                        kernel_size=args["kernel_size"],
                        strides=args["strides"],
                        se_ratio=args["se_ratio"],
                        activation=activation,
                        bn_momentum=bn_momentum,
                        survival_probability=drop_connect_rate * b / blocks,
                        name="block{}{}_".format(i + 1, chr(j + 97)),
                    ),
                    1: FusedMBConvBlock(
                        input_filters=args["input_filters"],
                        output_filters=args["output_filters"],
                        expand_ratio=args["expand_ratio"],
                        kernel_size=args["kernel_size"],
                        strides=args["strides"],
                        se_ratio=args["se_ratio"],
                        activation=activation,
                        bn_momentum=bn_momentum,
                        survival_probability=drop_connect_rate * b / blocks,
                        name="block{}{}_".format(i + 1, chr(j + 97)),
                    ),
                }[args["conv_type"]]

                x = block(x)
                b += 1

        # Build top
        top_filters = round_filters(
            filters=1280,
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor,
        )

        x = layers.Conv2D(
            filters=top_filters,
            kernel_size=1,
            strides=1,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            data_format="channels_last",
            use_bias=False,
            name="top_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=BN_AXIS,
            momentum=bn_momentum,
            name="top_bn",
        )(x)
        x = layers.Activation(activation=activation, name="top_activation")(x)

        inputs = img_input

        # Create model.
        super().__init__(inputs=inputs, outputs=x, **kwargs)

        self.include_rescaling = include_rescaling
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.default_size = default_size
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
        self.depth_divisor = depth_divisor
        self.min_depth = min_depth
        self.bn_momentum = bn_momentum
        self.activation = activation
        self.blocks_args = input_blocks_args
        self.model_name = model_name
        self.input_tensor = input_tensor

    def get_config(self):
        return {
            "include_rescaling": self.include_rescaling,
            "width_coefficient": self.width_coefficient,
            "depth_coefficient": self.depth_coefficient,
            "default_size": self.default_size,
            "dropout_rate": self.dropout_rate,
            "drop_connect_rate": self.drop_connect_rate,
            "depth_divisor": self.depth_divisor,
            "min_depth": self.min_depth,
            "bn_momentum": self.bn_momentum,
            "activation": self.activation,
            "blocks_args": self.blocks_args,
            "model_name": self.model_name,
            # Remove batch dimension from `input_shape`
            "input_shape": self.input_shape[1:],
            "input_tensor": self.input_tensor,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def EfficientNetV2B0(
    *,
    include_rescaling,
    input_shape=(None, None, 3),
    input_tensor=None,
    **kwargs,
):
    return EfficientNetV2(
        include_rescaling=include_rescaling,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=224,
        model_name="efficientnetv2-b0",
        input_shape=input_shape,
        input_tensor=input_tensor,
        **kwargs,
    )


def EfficientNetV2B1(
    *,
    include_rescaling,
    input_shape=(None, None, 3),
    input_tensor=None,
    **kwargs,
):
    return EfficientNetV2(
        include_rescaling=include_rescaling,
        width_coefficient=1.0,
        depth_coefficient=1.1,
        default_size=240,
        model_name="efficientnetv2-b1",
        input_shape=input_shape,
        input_tensor=input_tensor,
        **kwargs,
    )


def EfficientNetV2B2(
    *,
    include_rescaling,
    input_shape=(None, None, 3),
    input_tensor=None,
    **kwargs,
):
    return EfficientNetV2(
        include_rescaling=include_rescaling,
        width_coefficient=1.1,
        depth_coefficient=1.2,
        default_size=260,
        model_name="efficientnetv2-b2",
        input_shape=input_shape,
        input_tensor=input_tensor,
        **kwargs,
    )


def EfficientNetV2B3(
    *,
    include_rescaling,
    input_shape=(None, None, 3),
    input_tensor=None,
    **kwargs,
):
    return EfficientNetV2(
        include_rescaling=include_rescaling,
        width_coefficient=1.2,
        depth_coefficient=1.4,
        default_size=300,
        model_name="efficientnetv2-b3",
        input_shape=input_shape,
        input_tensor=input_tensor,
        **kwargs,
    )


def EfficientNetV2S(
    *,
    include_rescaling,
    input_shape=(None, None, 3),
    input_tensor=None,
    **kwargs,
):
    return EfficientNetV2(
        include_rescaling=include_rescaling,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=384,
        model_name="efficientnetv2-s",
        input_shape=input_shape,
        input_tensor=input_tensor,
        **kwargs,
    )


def EfficientNetV2M(
    *,
    include_rescaling,
    input_shape=(None, None, 3),
    input_tensor=None,
    **kwargs,
):
    return EfficientNetV2(
        include_rescaling=include_rescaling,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=480,
        model_name="efficientnetv2-m",
        input_shape=input_shape,
        input_tensor=input_tensor,
        **kwargs,
    )


def EfficientNetV2L(
    *,
    include_rescaling,
    input_shape=(None, None, 3),
    input_tensor=None,
    **kwargs,
):
    return EfficientNetV2(
        include_rescaling=include_rescaling,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        default_size=480,
        model_name="efficientnetv2-l",
        input_shape=input_shape,
        input_tensor=input_tensor,
        **kwargs,
    )


EfficientNetV2B0.__doc__ = BASE_DOCSTRING.format(name="EfficientNetV2B0")
EfficientNetV2B1.__doc__ = BASE_DOCSTRING.format(name="EfficientNetV2B1")
EfficientNetV2B2.__doc__ = BASE_DOCSTRING.format(name="EfficientNetV2B2")
EfficientNetV2B3.__doc__ = BASE_DOCSTRING.format(name="EfficientNetV2B3")
EfficientNetV2S.__doc__ = BASE_DOCSTRING.format(name="EfficientNetV2S")
EfficientNetV2M.__doc__ = BASE_DOCSTRING.format(name="EfficientNetV2M")
EfficientNetV2L.__doc__ = BASE_DOCSTRING.format(name="EfficientNetV2L")
