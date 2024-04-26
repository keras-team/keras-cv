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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.layers import FusedMBConvBlock
from keras_cv.src.layers import MBConvBlock
from keras_cv.src.models import utils
from keras_cv.src.models.backbones.backbone import Backbone
from keras_cv.src.models.backbones.efficientnet_v2.efficientnet_v2_backbone_presets import (  # noqa: E501
    backbone_presets,
)
from keras_cv.src.models.backbones.efficientnet_v2.efficientnet_v2_backbone_presets import (  # noqa: E501
    backbone_presets_with_weights,
)
from keras_cv.src.utils.python_utils import classproperty


@keras_cv_export("keras_cv.models.EfficientNetV2Backbone")
class EfficientNetV2Backbone(Backbone):
    """Instantiates the EfficientNetV2 architecture.

    Reference:
    - [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
      (ICML 2021)

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
            or 'fused' depending on the desired blocks.  FusedMBConvBlock is
            similar to MBConvBlock, but instead of using a depthwise convolution
            and a 1x1 output convolution blocks fused blocks use a single 3x3
            convolution block.
        skip_connection_dropout: float, dropout rate at skip connections.
        depth_divisor: integer, a unit of network width.
        min_depth: integer, minimum number of filters.
        activation: activation function to use between each convolutional layer.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `keras.layers.Input()`)
            to use as image input for the model.

    Example:
    ```python
    # Construct an EfficientNetV2 from a preset:
    efficientnet = keras_cv.models.EfficientNetV2Backbone.from_preset(
        "efficientnetv2_s"
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
    """  # noqa: E501

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
            x = keras.layers.Rescaling(scale=1 / 255.0)(x)

        # Build stem
        stem_filters = round_filters(
            filters=stackwise_input_filters[0],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor,
        )
        x = keras.layers.Conv2D(
            filters=stem_filters,
            kernel_size=3,
            strides=2,
            kernel_initializer=conv_kernel_initializer(),
            padding="same",
            use_bias=False,
            name="stem_conv",
        )(x)
        x = keras.layers.BatchNormalization(
            momentum=0.9,
            name="stem_bn",
        )(x)
        x = keras.layers.Activation(activation, name="stem_activation")(x)

        # Build blocks
        block_id = 0
        blocks = float(
            sum(num_repeats for num_repeats in stackwise_num_repeats)
        )

        pyramid_level_inputs = []
        for i in range(len(stackwise_kernel_sizes)):
            num_repeats = stackwise_num_repeats[i]
            input_filters = stackwise_input_filters[i]
            output_filters = stackwise_output_filters[i]

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
                repeats=num_repeats,
                depth_coefficient=depth_coefficient,
            )
            strides = stackwise_strides[i]
            squeeze_and_excite_ratio = stackwise_squeeze_and_excite_ratios[i]

            for j in range(repeats):
                # The first block needs to take care of stride and filter size
                # increase.
                if j > 0:
                    strides = 1
                    input_filters = output_filters

                if strides != 1:
                    pyramid_level_inputs.append(utils.get_tensor_input_name(x))

                # 97 is the start of the lowercase alphabet.
                letter_identifier = chr(j + 97)
                block = get_conv_constructor(stackwise_conv_types[i])(
                    input_filters=input_filters,
                    output_filters=output_filters,
                    expand_ratio=stackwise_expansion_ratios[i],
                    kernel_size=stackwise_kernel_sizes[i],
                    strides=strides,
                    se_ratio=squeeze_and_excite_ratio,
                    activation=activation,
                    survival_probability=skip_connection_dropout
                    * block_id
                    / blocks,
                    bn_momentum=0.9,
                    name="block{}{}_".format(i + 1, letter_identifier),
                )
                x = block(x)
                block_id += 1

        # Build top
        top_filters = round_filters(
            filters=1280,
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor,
        )

        x = keras.layers.Conv2D(
            filters=top_filters,
            kernel_size=1,
            strides=1,
            kernel_initializer=conv_kernel_initializer(),
            padding="same",
            data_format="channels_last",
            use_bias=False,
            name="top_conv",
        )(x)
        x = keras.layers.BatchNormalization(
            momentum=0.9,
            name="top_bn",
        )(x)
        x = keras.layers.Activation(
            activation=activation, name="top_activation"
        )(x)

        pyramid_level_inputs.append(utils.get_tensor_input_name(x))

        # Create model.
        super().__init__(inputs=img_input, outputs=x, **kwargs)

        self.include_rescaling = include_rescaling
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.skip_connection_dropout = skip_connection_dropout
        self.depth_divisor = depth_divisor
        self.min_depth = min_depth
        self.activation = activation
        self.input_tensor = input_tensor
        self.pyramid_level_inputs = {
            f"P{i + 1}": name for i, name in enumerate(pyramid_level_inputs)
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
