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


"""EfficientNet Lite backbone model.

Reference:
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
        (ICML 2019)
    - [Based on the original EfficientNet Lite's](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite)
"""  # noqa: E501

import copy
import math

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.models import utils
from keras_cv.src.models.backbones.backbone import Backbone
from keras_cv.src.models.backbones.efficientnet_lite.efficientnet_lite_backbone_presets import (  # noqa: E501
    backbone_presets,
)
from keras_cv.src.utils.python_utils import classproperty

BN_AXIS = 3


@keras_cv_export("keras_cv.models.EfficientNetLiteBackbone")
@keras.saving.register_keras_serializable(package="keras_cv.models")
class EfficientNetLiteBackbone(Backbone):
    """Instantiates the EfficientNetLite architecture using given scaling
    coefficients.

    Reference:
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
        (ICML 2019)
    - [Based on the original EfficientNet Lite's](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite)

    Args:
        include_rescaling: whether to rescale the inputs. If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections. The
            default value is set to 0.2.
        depth_divisor: integer, a unit of network width. The default value
            is set to 8.
        activation: activation function.
        input_shape: optional shape tuple,
            It should have exactly 3 inputs channels.
        input_tensor: optional Keras tensor (i.e. output of `keras.layers.Input()`)
            to use as image input for the model.

    Example:
    ```python
    # Construct an EfficientNetLite from a preset:
    efficientnet = models.EfficientNetLiteBackbone.from_preset(
        "efficientnetlite_b0"
    )
    images = np.ones((1, 256, 256, 3))
    outputs = efficientnet.predict(images)

    # Alternatively, you can also customize the EfficientNetLite architecture:
    model = EfficientNetLiteBackbone(
        stackwise_kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
        stackwise_num_repeats=[1, 2, 2, 3, 3, 4, 1],
        stackwise_input_filters=[32, 16, 24, 40, 80, 112, 192],
        stackwise_output_filters=[16, 24, 40, 80, 112, 192, 320],
        stackwise_expansion_ratios=[1, 6, 6, 6, 6, 6, 6],
        stackwise_strides=[1, 2, 2, 2, 1, 2, 1],
        width_coefficient=1.0,
        depth_coefficient=1.0,
        include_rescaling=False,
    )
    images = np.ones((1, 256, 256, 3))
    outputs = model.predict(images)
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
        stackwise_strides,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
        depth_divisor=8,
        input_shape=(None, None, 3),
        input_tensor=None,
        activation="relu6",
        **kwargs,
    ):
        img_input = utils.parse_model_inputs(input_shape, input_tensor)

        # Build stem
        x = img_input

        if include_rescaling:
            # Use common rescaling strategy across keras_cv
            x = keras.layers.Rescaling(1.0 / 255.0)(x)

        x = keras.layers.ZeroPadding2D(
            padding=utils.correct_pad_downsample(x, 3), name="stem_conv_pad"
        )(x)
        x = keras.layers.Conv2D(
            32,
            3,
            strides=2,
            padding="valid",
            use_bias=False,
            kernel_initializer=conv_kernel_initializer(),
            name="stem_conv",
        )(x)
        x = keras.layers.BatchNormalization(axis=BN_AXIS, name="stem_bn")(x)
        x = keras.layers.Activation(activation, name="stem_activation")(x)

        # Build blocks
        block_id = 0
        blocks = float(sum(stackwise_num_repeats))

        pyramid_level_inputs = []

        for i in range(len(stackwise_kernel_sizes)):
            num_repeats = stackwise_num_repeats[i]
            input_filters = stackwise_input_filters[i]
            output_filters = stackwise_output_filters[i]
            # Update block input and output filters based on depth multiplier.
            input_filters = round_filters(
                filters=input_filters,
                width_coefficient=width_coefficient,
                depth_divisor=depth_divisor,
            )
            output_filters = round_filters(
                filters=output_filters,
                width_coefficient=width_coefficient,
                depth_divisor=depth_divisor,
            )

            if i == 0 or i == (len(stackwise_kernel_sizes) - 1):
                repeats = num_repeats
            else:
                repeats = round_repeats(
                    repeats=num_repeats,
                    depth_coefficient=depth_coefficient,
                )
            strides = stackwise_strides[i]

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
                x = apply_efficient_net_lite_block(
                    inputs=x,
                    filters_in=input_filters,
                    filters_out=output_filters,
                    kernel_size=stackwise_kernel_sizes[i],
                    strides=strides,
                    expand_ratio=stackwise_expansion_ratios[i],
                    activation=activation,
                    dropout_rate=drop_connect_rate * block_id / blocks,
                    name="block{}{}_".format(i + 1, letter_identifier),
                )
                block_id += 1

        # Build top
        x = keras.layers.Conv2D(
            1280,
            1,
            padding="same",
            use_bias=False,
            kernel_initializer=conv_kernel_initializer(),
            name="top_conv",
        )(x)
        x = keras.layers.BatchNormalization(axis=BN_AXIS, name="top_bn")(x)
        x = keras.layers.Activation(activation, name="top_activation")(x)

        pyramid_level_inputs.append(utils.get_tensor_input_name(x))

        # Create model.
        super().__init__(inputs=img_input, outputs=x, **kwargs)

        self.include_rescaling = include_rescaling
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
        self.depth_divisor = depth_divisor
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
        self.stackwise_strides = stackwise_strides

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "include_rescaling": self.include_rescaling,
                "width_coefficient": self.width_coefficient,
                "depth_coefficient": self.depth_coefficient,
                "dropout_rate": self.dropout_rate,
                "drop_connect_rate": self.drop_connect_rate,
                "depth_divisor": self.depth_divisor,
                "activation": self.activation,
                "input_tensor": self.input_tensor,
                "input_shape": self.input_shape[1:],
                "stackwise_kernel_sizes": self.stackwise_kernel_sizes,
                "stackwise_num_repeats": self.stackwise_num_repeats,
                "stackwise_input_filters": self.stackwise_input_filters,
                "stackwise_output_filters": self.stackwise_output_filters,
                "stackwise_expansion_ratios": self.stackwise_expansion_ratios,
                "stackwise_strides": self.stackwise_strides,
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
    dropout_rate=0.0,
    name=None,
    filters_in=32,
    filters_out=16,
    kernel_size=3,
    strides=1,
    expand_ratio=1,
):
    """An inverted residual block, without SE phase.

    Args:
        inputs: input tensor.
        activation: activation function.
        dropout_rate: float between 0 and 1, fraction of the input units to drop.
        name: string, block label.
        filters_in: integer, the number of input filters.
        filters_out: integer, the number of output filters.
        kernel_size: integer, the dimension of the convolution window.
        strides: integer, the stride of the convolution.
        expand_ratio: integer, scaling coefficient for the input filters.

    Returns:
        output tensor for the block.
    """  # noqa: E501
    if name is None:
        name = f"block_{keras.backend.get_uid('block_')}_"

    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = keras.layers.Conv2D(
            filters,
            1,
            padding="same",
            use_bias=False,
            kernel_initializer=conv_kernel_initializer(),
            name=name + "expand_conv",
        )(inputs)
        x = keras.layers.BatchNormalization(
            axis=BN_AXIS, name=name + "expand_bn"
        )(x)
        x = keras.layers.Activation(
            activation, name=name + "expand_activation"
        )(x)
    else:
        x = inputs

    # Depthwise Convolution
    if strides == 2:
        x = keras.layers.ZeroPadding2D(
            padding=utils.correct_pad_downsample(x, kernel_size),
            name=name + "dwconv_pad",
        )(x)
        conv_pad = "valid"
    else:
        conv_pad = "same"
    x = keras.layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        padding=conv_pad,
        use_bias=False,
        depthwise_initializer=conv_kernel_initializer(),
        name=name + "dwconv",
    )(x)
    x = keras.layers.BatchNormalization(axis=BN_AXIS, name=name + "bn")(x)
    x = keras.layers.Activation(activation, name=name + "activation")(x)

    # Output phase
    x = keras.layers.Conv2D(
        filters_out,
        1,
        padding="same",
        use_bias=False,
        kernel_initializer=conv_kernel_initializer(),
        name=name + "project_conv",
    )(x)
    x = keras.layers.BatchNormalization(axis=BN_AXIS, name=name + "project_bn")(
        x
    )
    if strides == 1 and filters_in == filters_out:
        if dropout_rate > 0:
            x = keras.layers.Dropout(
                dropout_rate, noise_shape=(None, 1, 1, 1), name=name + "drop"
            )(x)
        x = keras.layers.Add(name=name + "add")([x, inputs])
    return x
