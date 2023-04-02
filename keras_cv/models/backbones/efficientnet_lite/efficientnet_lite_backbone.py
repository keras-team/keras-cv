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


"""EfficientNet Lite models for Keras.

Reference:
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
        https://arxiv.org/abs/1905.11946) (ICML 2019)
    - [Based on the original EfficientNet Lite's](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite)
"""

import copy
import math

from keras_cv.models import utils
# from keras_cv.models.weights import parse_weights
from keras_cv.models.backbones.backbone import Backbone
from keras_cv.models.backbones.efficientnet_lite.efficientnet_lite_backbone_presets import (
    backbone_presets, 
)
from keras_cv.utils.python_utils import classproperty
from tensorflow import keras
from tensorflow.keras import backend, layers

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


def apply_efficient_net_lite_block(
    inputs,
    activation="relu6",
    drop_rate=0.0,
    name=None,
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
    if name is None:
        name = f"block_{backend.get_uid('block_')}_"

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


@keras.utils.register_keras_serializable(package="keras_cv.models")
class EfficientNetLiteBackbone(Backbone):
    """Instantiates the EfficientNetLite architecture.

    Args:
        include_rescaling: whether to Rescale the inputs. If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        default_size: integer, default input image size.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: integer, a unit of network width.
        activation: activation function.
        blocks_args: list of dicts, parameters to construct block modules.
        input_shape: optional shape tuple,
            It should have exactly 3 inputs channels.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.


        Raises:
            ValueError: if `blocks_args` is invalid.
    """

    def __init__(
        self,
        *,
        include_rescaling,
        width_coefficient,
        depth_coefficient,
        default_size,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
        depth_divisor=8,
        activation="relu6",
        blocks_args=None,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        if blocks_args is None:
            blocks_args = DEFAULT_BLOCKS_ARGS
        if not isinstance(blocks_args, list):
            raise ValueError(
                "The `blocks_args` argument should be either `None` or valid"
                "list of dicts for building blocks. "
                f"Received: blocks_args={blocks_args}"
            )
        intact_blocks_args = copy.deepcopy(blocks_args)  # for configs
        blocks_args = copy.deepcopy(blocks_args)

        img_input = utils.parse_model_inputs(input_shape, input_tensor)

        # Build stem
        x = img_input

        if include_rescaling:
            # Use common rescaling strategy across keras_cv
            x = layers.Rescaling(1.0 / 255.0)(x)

        x = layers.ZeroPadding2D(
            padding=correct_pad(x, 3), name="stem_conv_pad"
        )(x)
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
        b = 0
        blocks = float(sum(args["repeats"] for args in blocks_args))

        for i, args in enumerate(blocks_args):
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
                    repeats=args.pop("repeats"),
                    depth_coefficient=depth_coefficient,
                )

            for j in range(repeats):
                # The first block needs to take care of stride and filter size
                # increase.
                if j > 0:
                    args["strides"] = 1
                    args["filters_in"] = args["filters_out"]
                x = apply_efficient_net_lite_block(
                    x,
                    activation=activation,
                    drop_rate=drop_connect_rate * b / blocks,
                    name="block{}{}_".format(i + 1, chr(j + 97)),
                    **args,
                )

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

        inputs = img_input

        # Create model.
        super().__init__(inputs=inputs, outputs=x, **kwargs)

        # All references to `self` below this line
        self.include_rescaling = include_rescaling
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.default_size = default_size
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
        self.depth_divisor = depth_divisor
        self.activation = activation
        self.blocks_args = intact_blocks_args
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "include_rescaling": self.include_rescaling,
                "width_coefficient": self.width_coefficient,
                "depth_coefficient": self.depth_coefficient,
                "default_size": self.default_size,
                "dropout_rate": self.dropout_rate,
                "drop_connect_rate": self.drop_connect_rate,
                "depth_divisor": self.depth_divisor,
                "activation": self.activation,
                "blocks_args": self.blocks_args,
                # Remove batch dimension from `input_shape`
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
            }
        )
        return config

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy(backbone_presets)


ALIAS_DOCSTRING = """EfficientNetLiteBackbone model with {width_coefficient} width coefficient
    and {depth_coefficient} depth coefficient.

    Reference:
        - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
        https://arxiv.org/abs/1905.11946) (ICML 2019)


    For image classification use cases, see
    [this page for detailed examples]
    (https://keras.io/api/applications/#usage-examples-for-image-classification-models).

    For transfer learning use cases, make sure to read the [guide to transfer
        learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        include_rescaling: bool, whether or not to Rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        num_classes: optional int, number of classes to classify images into (only
            to be specified if `include_top` is `True`).
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

"""


class EfficientNetLiteB0Backbone(EfficientNetLiteBackbone):
    def __new__(
        cls,
        include_rescaling,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return EfficientNetLiteBackbone.from_preset(
            "efficientnetliteb0", **kwargs
        )

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}


class EfficientNetLiteB1Backbone(EfficientNetLiteBackbone):
    def __new__(
        cls,
        include_rescaling,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return EfficientNetLiteBackbone.from_preset(
            "efficientnetliteb1", **kwargs
        )

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}


class EfficientNetLiteB2Backbone(EfficientNetLiteBackbone):
    def __new__(
        cls,
        include_rescaling,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return EfficientNetLiteBackbone.from_preset(
            "efficientnetliteb2", **kwargs
        )

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}


class EfficientNetLiteB3Backbone(EfficientNetLiteBackbone):
    def __new__(
        cls,
        include_rescaling,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return EfficientNetLiteBackbone.from_preset(
            "efficientnetliteb3", **kwargs
        )

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}


class EfficientNetLiteB4Backbone(EfficientNetLiteBackbone):
    def __new__(
        cls,
        include_rescaling,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return EfficientNetLiteBackbone.from_preset(
            "efficientnetliteb4", **kwargs
        )

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}


setattr(
    EfficientNetLiteB0Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(
        name="B0", width_coefficient="1.0", depth_coefficient="1.0"
    ),
)

setattr(
    EfficientNetLiteB1Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(
        name="B1", width_coefficient="1.0", depth_coefficient="1.1"
    ),
)

setattr(
    EfficientNetLiteB2Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(
        name="B2", width_coefficient="1.1", depth_coefficient="1.2"
    ),
)

setattr(
    EfficientNetLiteB3Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(
        name="B3", width_coefficient="1.2", depth_coefficient="1.4"
    ),
)

setattr(
    EfficientNetLiteB4Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(
        name="B4", width_coefficient="1.4", depth_coefficient="1.8"
    ),
)
