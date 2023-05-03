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


@keras.utils.register_keras_serializable(package="keras_cv.models")
class EfficientNetV2Backbone(Backbone):
    """Instantiates the EfficientNetV2 architecture.

    Args:
        include_rescaling: bool, whether to rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        stackwise_kernel_sizes:  list of ints, the kernel sizes used for each
            conv block.
        stackwise_num_repeats: list of ints, number of times to repeat each
            conv block.
        stackwise_input_filters: list of ints, number of input filters for
            each conv block.
        stackwise_output_filters: list of ints, number of output filters for
            each stack in the conv blocks model.
        stackwise_expansion_ratios: list of floats, expand ratio passed to the
            squeeze and excitation blocks.
        stackwise_squeeze_and_excite_ratios: list of ints, the squeeze and
            excite ratios passed to the squeeze and excitation blocks.
        stackwise_strides: list of ints, stackwise_strides for each conv block.
        stackwise_conv_types: list of strings.  Each value is either 'unfused'
            or 'fused' depending on the desired blocks.
        skip_connection_dropout: float, dropout rate at skip connections.
        depth_divisor: integer, a unit of network width.
        min_depth: integer, minimum number of filters.
        activation: activation function to use between each convolutional layer.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    Usage:
    ```python
    # Construct an EfficientNetV2 from a preset:
    efficientnet = keras_cv.models.EfficientNetV2Backbone.from_preset(
        "efficientnetv2-s"
    )
    images = tf.ones((1, 256, 256, 3))
    outputs = efficientnet.predict(images)

    # Alternatively, you can also customize the EfficientNetV2 architecture:
    model = EfficientNetV2Backbone(
        stackwise_kernel_sizes=[3, 3, 3, 3, 3, 3],
        stackwise_num_repeats=[2, 4, 4, 6, 9, 15],
        stackwise_input_filters=[24, 24, 48, 64, 128, 160],
        stackwise_output_filters=[24, 48, 64, 128, 160, 256],
        stackwise_expansion_ratios=[1, 4, 4, 4, 6, 6],
        stackwise_squeeze_and_excite_ratios=[0.0, 0.0, 0, 0.25, 0.25, 0.25],
        stackwise_strides=[1, 2, 2, 2, 1, 2],
        stackwise_conv_types=[
            "fused",
            "fused",
            "fused",
            "unfused",
            "unfused",
            "unfused",
        ],
        width_coefficient=1.0,
        depth_coefficient=1.0,
        include_rescaling=False,
    )
    images = tf.ones((1, 256, 256, 3))
    outputs = efficientnet.predict(images)
    ```
    """  # noqa: E502

    def __init__(
        self,
        *,
        include_rescaling,
        width_coefficient,
        depth_coefficient,
        stackwise_kernel_sizes,
        stackwise_num_repeats,
        stackwise_input_filters,
        stackwise_output_filters,
        stackwise_expansion_ratios,
        stackwise_squeeze_and_excite_ratios,
        stackwise_strides,
        stackwise_conv_types,
        skip_connection_dropout=0.2,
        depth_divisor=8,
        min_depth=8,
        activation="swish",
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Determine proper input shape
        img_input = utils.parse_model_inputs(input_shape, input_tensor)

        x = img_input

        if include_rescaling:
            x = layers.Rescaling(scale=1 / 255.0)(x)

        # Build stem
        stem_filters = round_filters(
            filters=stackwise_input_filters[0],
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
        b = 0
        blocks = float(sum(num_repeat for num_repeat in stackwise_num_repeats))

        pyramid_level_inputs = []
        for i in range(len(stackwise_kernel_sizes)):
            kernel_size = stackwise_kernel_sizes[i]
            num_repeat = stackwise_num_repeats[i]
            input_filters = stackwise_input_filters[i]
            output_filters = stackwise_output_filters[i]
            expand_ratio = stackwise_expansion_ratios[i]
            squeeze_and_excite_ratio = stackwise_squeeze_and_excite_ratios[i]
            strides = stackwise_strides[i]
            conv_type = stackwise_conv_types[i]

            # Update block input and output filters based on depth multiplier.
            input_filters = round_filters(
                filters=input_filters,
                width_coefficient=width_coefficient,
                min_depth=min_depth,
                depth_divisor=depth_divisor,
            )
            output_filters = round_filters(
                filters=output_filters,
                width_coefficient=width_coefficient,
                min_depth=min_depth,
                depth_divisor=depth_divisor,
            )

            repeats = round_repeats(
                repeats=num_repeat,
                depth_coefficient=depth_coefficient,
            )
            for j in range(repeats):
                # The first block needs to take care of stride and filter size
                # increase.
                if j > 0:
                    strides = 1
                    input_filters = output_filters

                if strides != 1:
                    pyramid_level_inputs.append(x.node.layer.name)

                block = get_block_conv(
                    conv_type=conv_type,
                    input_filters=input_filters,
                    output_filters=output_filters,
                    expand_ratio=expand_ratio,
                    kernel_size=kernel_size,
                    strides=strides,
                    squeeze_and_excite_ratio=squeeze_and_excite_ratio,
                    activation=activation,
                    survival_probability=skip_connection_dropout * b / blocks,
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
        self.skip_connection_dropout = skip_connection_dropout
        self.depth_divisor = depth_divisor
        self.min_depth = min_depth
        self.activation = activation
        self.input_tensor = input_tensor
        self.pyramid_level_inputs = {
            i + 1: name for i, name in enumerate(pyramid_level_inputs)
        }
        self.stackwise_kernel_sizes = stackwise_kernel_sizes
        self.stackwise_num_repeats = stackwise_num_repeats
        self.stackwise_input_filters = stackwise_input_filters
        self.stackwise_output_filters = stackwise_output_filters
        self.stackwise_expansion_ratios = stackwise_expansion_ratios
        self.stackwise_squeeze_and_excite_ratios = (
            stackwise_squeeze_and_excite_ratios
        )
        self.stackwise_strides = stackwise_strides
        self.stackwise_conv_types = stackwise_conv_types

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "include_rescaling": self.include_rescaling,
                "width_coefficient": self.width_coefficient,
                "depth_coefficient": self.depth_coefficient,
                "skip_connection_dropout": self.skip_connection_dropout,
                "depth_divisor": self.depth_divisor,
                "min_depth": self.min_depth,
                "activation": self.activation,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "stackwise_kernel_sizes": self.stackwise_kernel_sizes,
                "stackwise_num_repeats": self.stackwise_num_repeats,
                "stackwise_input_filters": self.stackwise_input_filters,
                "stackwise_output_filters": self.stackwise_output_filters,
                "stackwise_expansion_ratios": self.stackwise_expansion_ratios,
                "stackwise_squeeze_and_excite_ratios": self.stackwise_squeeze_and_excite_ratios,  # noqa: E501
                "stackwise_strides": self.stackwise_strides,
                "stackwise_conv_types": self.stackwise_conv_types,
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


def get_conv_constructor(conv_type):
    if conv_type == "unfused":
        return MBConvBlock
    elif conv_type == "fused":
        return FusedMBConvBlock
    else:
        raise ValueError(
            "Expected `conv_type` to be "
            "one of 'unfused', 'fused', but got "
            f"`conv_type={conv_type}`"
        )


def get_block_conv(
    conv_type,
    input_filters,
    output_filters,
    expand_ratio,
    kernel_size,
    strides,
    squeeze_and_excite_ratio,
    activation,
    survival_probability,
    name,
):
    return get_conv_constructor(conv_type)(
        input_filters=input_filters,
        output_filters=output_filters,
        expand_ratio=expand_ratio,
        kernel_size=kernel_size,
        strides=strides,
        se_ratio=squeeze_and_excite_ratio,
        activation=activation,
        bn_momentum=0.9,
        survival_probability=survival_probability,
        name=name,
    )


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
        return {
            "efficientnetv2-s": copy.deepcopy(
                backbone_presets["efficientnetv2-s"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {
            "efficientnetv2-s_imagenet": copy.deepcopy(
                backbone_presets["efficientnetv2-s_imagenet"]
            ),
        }


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
        return {
            "efficientnetv2-m": copy.deepcopy(
                backbone_presets["efficientnetv2-m"]
            ),
        }

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
        return {
            "efficientnetv2-l": copy.deepcopy(
                backbone_presets["efficientnetv2-l"]
            ),
        }

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
        return {
            "efficientnetv2-b0": copy.deepcopy(
                backbone_presets["efficientnetv2-b0"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {
            "efficientnetv2-b0_imagenet": copy.deepcopy(
                backbone_presets["efficientnetv2-b0_imagenet"]
            ),
        }


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
        return {
            "efficientnetv2-b1": copy.deepcopy(
                backbone_presets["efficientnetv2-b1"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {
            "efficientnetv2-b1_imagenet": copy.deepcopy(
                backbone_presets["efficientnetv2-b1_imagenet"]
            ),
        }


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
        return {
            "efficientnetv2-b2": copy.deepcopy(
                backbone_presets["efficientnetv2-b2"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {
            "efficientnetv2-b2_imagenet": copy.deepcopy(
                backbone_presets["efficientnetv2-b2_imagenet"]
            ),
        }


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
