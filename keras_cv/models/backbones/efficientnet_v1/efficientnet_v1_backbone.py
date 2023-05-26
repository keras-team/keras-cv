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

"""EfficientNet V1 models for Keras.

Reference:
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
        (ICML 2019)
    - [Based on the original keras.applications EfficientNet](https://github.com/keras-team/keras/blob/master/keras/applications/efficientnet.py)
"""  # noqa: E501

import copy
import math

from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers

from keras_cv.models import utils
from keras_cv.models.backbones.backbone import Backbone
from keras_cv.models.backbones.efficientnet_v1.efficientnet_v1_backbone_presets import (  # noqa: E501
    backbone_presets,
)
from keras_cv.utils.python_utils import classproperty

BASE_DOCSTRING = """Instantiates the {name} architecture.

    Reference:
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
    (ICML 2019) # noqa: E501

    This class represents a Keras image classification model.

    For image classification use cases, see
    [this page for detailed examples](https://keras.io/api/applications/#usage-examples-for-image-classification-models). # noqa: E501

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/). # noqa: E501

    Args:
        include_rescaling: bool, whether to rescale the inputs. If set to
            True, inputs will be passed through a `Rescaling(1/255.0)` layer.
        input_shape: tuple, Optional shape tuple. It should have exactly 3
            inputs channels.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to
            use as image input for the model.
"""


@keras.utils.register_keras_serializable(package="keras_cv.models")
class EfficientNetV1Backbone(Backbone):
    """This class represents a Keras EfficientNet architecture.
    Args:
        include_rescaling: bool, whether to rescale the inputs. If set to
            True, inputs will be passed through a `Rescaling(1/255.0)` layer.
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        default_size: integer, default input image size.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: integer, a unit of network width.
        activation: activation function.
        input_shape: optional shape tuple, it should have exactly 3 input
            channels.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to
            use as image input for the model.
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
        activation="swish",
        input_shape=(None, None, 3),
        input_tensor=None,
        stackwise_kernel_sizes,
        stackwise_num_repeats,
        stackwise_input_filters,
        stackwise_output_filters,
        stackwise_expansion_ratios,
        stackwise_id_skip,
        stackwise_strides,
        stackwise_squeeze_and_excite_ratios,
        **kwargs,
    ):
        img_input = utils.parse_model_inputs(input_shape, input_tensor)

        x = img_input

        if include_rescaling:
            # Use common rescaling strategy across keras_cv
            x = layers.Rescaling(1.0 / 255.0)(x)

        x = layers.ZeroPadding2D(
            padding=correct_pad(x, 3), name="stem_conv_pad"
        )(x)

        # Build stem
        stem_filters = round_filters(
            filters=stackwise_input_filters[0],
            width_coefficient=width_coefficient,
            divisor=depth_divisor,
        )
        x = layers.Conv2D(
            filters=stem_filters,
            kernel_size=3,
            strides=2,
            padding="valid",
            use_bias=False,
            kernel_initializer=conv_kernel_initializer(),
            name="stem_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=3,
            name="stem_bn",
        )(x)
        x = layers.Activation(activation, name="stem_activation")(x)

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
                divisor=depth_divisor,
            )
            output_filters = round_filters(
                filters=output_filters,
                width_coefficient=width_coefficient,
                divisor=depth_divisor,
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
                    pyramid_level_inputs.append(x.node.layer.name)

                # 97 is the start of the lowercase alphabet.
                letter_identifier = chr(j + 97)
                x = apply_efficientnet_block(
                    inputs=x,
                    filters_in=input_filters,
                    filters_out=output_filters,
                    kernel_size=stackwise_kernel_sizes[i],
                    strides=strides,
                    expand_ratio=stackwise_expansion_ratios[i],
                    se_ratio=squeeze_and_excite_ratio,
                    id_skip=True,
                    activation=activation,
                    drop_rate=drop_connect_rate * block_id / blocks,
                    name="block{}{}_".format(i + 1, letter_identifier),
                )
                block_id += 1

        # Build top
        top_filters = round_filters(
            filters=1280,
            width_coefficient=width_coefficient,
            divisor=depth_divisor,
        )

        x = layers.Conv2D(
            filters=top_filters,
            kernel_size=1,
            padding="same",
            strides=1,
            kernel_initializer=conv_kernel_initializer(),
            use_bias=False,
            name="top_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=3,
            name="top_bn",
        )(x)
        x = layers.Activation(activation=activation, name="top_activation")(x)

        pyramid_level_inputs.append(x.node.layer.name)

        # Create model.
        super().__init__(inputs=img_input, outputs=x, **kwargs)

        self.include_rescaling = include_rescaling
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.default_size = default_size
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
        self.depth_divisor = depth_divisor
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
        self.stackwise_id_skip = stackwise_id_skip
        self.stackwise_strides = stackwise_strides
        self.stackwise_squeeze_and_excite_ratios = (
            stackwise_squeeze_and_excite_ratios
        )

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
                "input_tensor": self.input_tensor,
                "input_shape": self.input_shape[1:],
                "trainable": self.trainable,
                "stackwise_kernel_sizes": self.stackwise_kernel_sizes,
                "stackwise_num_repeats": self.stackwise_num_repeats,
                "stackwise_input_filters": self.stackwise_input_filters,
                "stackwise_output_filters": self.stackwise_output_filters,
                "stackwise_expansion_ratios": self.stackwise_expansion_ratios,
                "stackwise_id_skip": self.stackwise_id_skip,
                "stackwise_strides": self.stackwise_strides,
                "stackwise_squeeze_and_excite_ratios": (
                    self.stackwise_squeeze_and_excite_ratios
                ),
            }
        )
        return config

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy(backbone_presets)


def conv_kernel_initializer(scale=2.0):
    return keras.initializers.VarianceScaling(
        scale=scale, mode="fan_out", distribution="truncated_normal"
    )


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
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


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
        x = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer=conv_kernel_initializer(),
            name="expand_conv",
        )(inputs)
        x = layers.BatchNormalization(
            axis=3,
            name="expand_bn",
        )(x)
        x = layers.Activation(activation, name="expand_activation")(x)
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

    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding=conv_pad,
        use_bias=False,
        kernel_initializer=conv_kernel_initializer(),
        name="depthwise_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=3,
        name="depthwise_bn",
    )(x)
    x = layers.Activation(activation, name="depthwise_activation")(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D(name="_se_squeeze")(x)
        se_shape = (1, 1, filters)
        se = layers.Reshape(se_shape, name="_se_reshape")(se)
        se = layers.Conv2D(
            filters_se,
            1,
            padding="same",
            activation=activation,
            kernel_initializer=conv_kernel_initializer(),
            name="_se_reduce",
        )(se)
        se = layers.Conv2D(
            filters,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=conv_kernel_initializer(),
            name="_se_expand",
        )(se)
        x = layers.multiply([x, se], name="_se_excite")

    # Output phase
    x = layers.Conv2D(
        filters=filters_out,
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer=conv_kernel_initializer(),
        name="output_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=3,
        name="output_bn",
    )(x)
    x = layers.Activation(activation, name="output_activation")(x)

    if id_skip and strides == 1 and filters_in == filters_out:
        if drop_rate > 0:
            x = layers.Dropout(
                drop_rate,
                noise_shape=(None, 1, 1, 1),
                name=name + "_drop",
            )(x)
        x = layers.add([x, inputs], name=name + "_add")

    return x
