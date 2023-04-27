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
from keras_cv.models.backbones.backbone import Backbone
from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_backbone_presets import (  # noqa: E501
    backbone_presets,
)
from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_backbone_presets import (  # noqa: E501
    backbone_presets_with_weights,
)
from keras_cv.utils.python_utils import classproperty


def conv_kernel_initializer(scale=2.0):
    return keras.initializers.VarianceScaling(
        scale=scale, mode="fan_out", distribution="truncated_normal"
    )


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


def get_block_conv(args, activation, survival_probability, name):
    # Determine which conv type to use:
    if args["conv_type"] == "mb_conv":
        return MBConvBlock(
            input_filters=args["input_filters"],
            output_filters=args["output_filters"],
            expand_ratio=args["expand_ratio"],
            kernel_size=args["kernel_size"],
            strides=args["strides"],
            se_ratio=args["se_ratio"],
            activation=activation,
            bn_momentum=0.9,
            survival_probability=survival_probability,
            name=name,
        )
    elif args["conv_type"] == "fused_mb_conv":
        return FusedMBConvBlock(
            input_filters=args["input_filters"],
            output_filters=args["output_filters"],
            expand_ratio=args["expand_ratio"],
            kernel_size=args["kernel_size"],
            strides=args["strides"],
            se_ratio=args["se_ratio"],
            activation=activation,
            bn_momentum=0.9,
            survival_probability=survival_probability,
            name=name,
        )
    raise ValueError(
        "Expected `block_args['conv_type']` to be "
        "one of 'mb_conv', 'fused_mb_conv', but got "
        f"`block_args['conv_type']={args['conv_type']}`"
    )


@keras.utils.register_keras_serializable(package="keras_cv.models")
class EfficientNetV2Backbone(Backbone):
    """Instantiates the EfficientNetV2 architecture.

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
        activation: activation function.
        block_args: list of dicts, parameters to construct block modules.
        model_name: string, model name.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    Usage:
    ```python
    efficientnet = keras_cv.models.EfficientNetV2Backbone.from_preset(
        "efficientnetv2-s"
    )
    images = tf.ones((1, 256, 256, 3))
    outputs = efficientnet.predict(images)
    ```
    """

    def __init__(
        self,
        *,
        include_rescaling,
        width_coefficient,
        depth_coefficient,
        default_size,
        block_args,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
        depth_divisor=8,
        min_depth=8,
        activation="swish",
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        input_block_args = copy.deepcopy(block_args)

        # Determine proper input shape
        img_input = utils.parse_model_inputs(input_shape, input_tensor)

        x = img_input

        if include_rescaling:
            x = layers.Rescaling(scale=1 / 255.0)(x)

        # Build stem
        stem_filters = round_filters(
            filters=block_args[0]["input_filters"],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor,
        )
        x = layers.Conv2D(
            filters=stem_filters,
            kernel_size=3,
            strides=2,
            kernel_initializer=conv_kernel_initializer(),
            padding="same",
            use_bias=False,
            name="stem_conv",
        )(x)
        x = layers.BatchNormalization(
            momentum=0.9,
            name="stem_bn",
        )(x)
        x = layers.Activation(activation, name="stem_activation")(x)

        # Build blocks
        block_args = copy.deepcopy(block_args)
        b = 0
        blocks = float(sum(args["num_repeat"] for args in block_args))

        pyramid_level_inputs = []
        for i, args in enumerate(block_args):
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

                if args["strides"] != 1:
                    pyramid_level_inputs.append(x.node.layer.name)

                block = get_block_conv(
                    args,
                    activation=activation,
                    survival_probability=drop_connect_rate * b / blocks,
                    name="block{}{}_".format(i + 1, chr(j + 97)),
                )
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
            kernel_initializer=conv_kernel_initializer(),
            padding="same",
            data_format="channels_last",
            use_bias=False,
            name="top_conv",
        )(x)
        x = layers.BatchNormalization(
            momentum=0.9,
            name="top_bn",
        )(x)
        x = layers.Activation(activation=activation, name="top_activation")(x)

        pyramid_level_inputs.append(x.node.layer.name)
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
        self.activation = activation
        self.block_args = input_block_args
        self.input_tensor = input_tensor
        self.pyramid_level_inputs = {
            i + 1: name for i, name in enumerate(pyramid_level_inputs)
        }

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
                "min_depth": self.min_depth,
                "activation": self.activation,
                "block_args": self.block_args,
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

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return copy.deepcopy(backbone_presets_with_weights)


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


class EfficientNetV2SBackbone(EfficientNetV2Backbone):
    def __new__(
        cls,
        include_rescaling=True,
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
        return EfficientNetV2Backbone.from_preset("efficientnetv2-s", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class EfficientNetV2MBackbone(EfficientNetV2Backbone):
    def __new__(
        cls,
        include_rescaling=True,
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
        return EfficientNetV2Backbone.from_preset("efficientnetv2-m", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class EfficientNetV2LBackbone(EfficientNetV2Backbone):
    def __new__(
        cls,
        include_rescaling=True,
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
        return EfficientNetV2Backbone.from_preset("efficientnetv2-l", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class EfficientNetV2B0Backbone(EfficientNetV2Backbone):
    def __new__(
        cls,
        include_rescaling=True,
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
        return EfficientNetV2Backbone.from_preset("efficientnetv2-b0", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class EfficientNetV2B1Backbone(EfficientNetV2Backbone):
    def __new__(
        cls,
        include_rescaling=True,
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
        return EfficientNetV2Backbone.from_preset("efficientnetv2-b1", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class EfficientNetV2B2Backbone(EfficientNetV2Backbone):
    def __new__(
        cls,
        include_rescaling=True,
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
        return EfficientNetV2Backbone.from_preset("efficientnetv2-b2", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class EfficientNetV2B3Backbone(EfficientNetV2Backbone):
    def __new__(
        cls,
        include_rescaling=True,
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
        return EfficientNetV2Backbone.from_preset("efficientnetv2-b3", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


EfficientNetV2B0Backbone.__doc__ = BASE_DOCSTRING.format(
    name="EfficientNetV2B0"
)
EfficientNetV2B1Backbone.__doc__ = BASE_DOCSTRING.format(
    name="EfficientNetV2B1"
)
EfficientNetV2B2Backbone.__doc__ = BASE_DOCSTRING.format(
    name="EfficientNetV2B2"
)
EfficientNetV2B3Backbone.__doc__ = BASE_DOCSTRING.format(
    name="EfficientNetV2B3"
)
EfficientNetV2SBackbone.__doc__ = BASE_DOCSTRING.format(name="EfficientNetV2S")
EfficientNetV2MBackbone.__doc__ = BASE_DOCSTRING.format(name="EfficientNetV2M")
EfficientNetV2LBackbone.__doc__ = BASE_DOCSTRING.format(name="EfficientNetV2L")
