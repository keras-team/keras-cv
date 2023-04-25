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

"""MobileNet v3 backbone model.

References:
    - [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf)
    (ICCV 2019)
    - [Based on the original keras.applications MobileNetv3](https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet_v3.py)
"""  # noqa: E501

import copy

from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras.utils import custom_object_scope

from keras_cv import layers as cv_layers
from keras_cv.models import utils
from keras_cv.models.backbones.backbone import Backbone
from keras_cv.models.backbones.mobilenet_v3.mobilenet_v3_backbone_presets import (  # noqa: E501
    backbone_presets,
)
from keras_cv.utils.python_utils import classproperty

CHANNEL_AXIS = -1
BN_EPSILON = 1e-3
BN_MOMENTUM = 0.999


def adjust_channels(x, divisor=8, min_value=None):
    """Ensure that all layers have a channel number divisible by the `divisor`.

    Args:
        x: integer, input value.
        divisor: integer, the value by which a channel number should be
            divisible, defaults to 8.
        min_value: float, optional minimum value for the new tensor. If None,
            defaults to value of divisor.

    Returns:
        the updated input scalar.
    """

    if min_value is None:
        min_value = divisor

    new_x = max(min_value, int(x + divisor / 2) // divisor * divisor)

    # make sure that round down does not go down by more than 10%.
    if new_x < 0.9 * x:
        new_x += divisor
    return new_x


def apply_hard_sigmoid(x):
    """Applies the hard sigmoid activation to x and returns the result.

    Args:
        x: input tensor
    Returns:
        the updated input tensor.
    """

    activation = layers.ReLU(6.0)

    return activation(x + 3.0) * (1.0 / 6.0)


def apply_hard_swish(x):
    """Applies the hard swish activation to x and returns the result.

    Args:
        x: input tensor

    Returns:
        the updated input tensor.
    """

    return layers.Multiply()([x, apply_hard_sigmoid(x)])


def apply_inverted_res_block(
    x,
    expansion,
    filters,
    kernel_size,
    stride,
    se_ratio,
    activation,
    expansion_index,
):
    """An Inverted Residual Block.

    Args:
        x: input tensor.
        expansion: integer, the expansion ratio, multiplied with infilters to
            get the minimum value passed to adjust_channels.
        filters: integer, number of filters for convolution layer.
        kernel_size: integer, the kernel size for DepthWise Convolutions.
        stride: integer, the stride length for DepthWise Convolutions.
        se_ratio: float, ratio for bottleneck filters. Number of bottleneck
            filters = filters * se_ratio.
        activation: the activation layer to use.
        expansion_index: integer, a unique identification if you want to use
            expanded convolutions. If greater than 0, an additional Conv+BN
            layer is added after the expanded convolutional layer.

    Returns:
        the updated input tensor.
    """
    if isinstance(activation, str):
        if activation == "hard_swish":
            activation = apply_hard_swish
        else:
            activation = keras.activations.get(activation)

    shortcut = x
    prefix = "expanded_conv/"
    infilters = backend.int_shape(x)[CHANNEL_AXIS]

    if expansion_index > 0:
        prefix = f"expanded_conv_{expansion_index}/"

        x = layers.Conv2D(
            adjust_channels(infilters * expansion),
            kernel_size=1,
            padding="same",
            use_bias=False,
            name=prefix + "expand",
        )(x)
        x = layers.BatchNormalization(
            axis=CHANNEL_AXIS,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name=prefix + "expand/BatchNorm",
        )(x)
        x = activation(x)

    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding="same" if stride == 1 else "valid",
        use_bias=False,
        name=prefix + "depthwise",
    )(x)
    x = layers.BatchNormalization(
        axis=CHANNEL_AXIS,
        epsilon=BN_EPSILON,
        momentum=BN_MOMENTUM,
        name=prefix + "depthwise/BatchNorm",
    )(x)
    x = activation(x)

    if se_ratio:
        with custom_object_scope({"hard_sigmoid": apply_hard_sigmoid}):
            x = cv_layers.SqueezeAndExcite2D(
                filters=adjust_channels(infilters * expansion),
                ratio=se_ratio,
                squeeze_activation="relu",
                excite_activation="hard_sigmoid",
            )(x)

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name=prefix + "project",
    )(x)
    x = layers.BatchNormalization(
        axis=CHANNEL_AXIS,
        epsilon=BN_EPSILON,
        momentum=BN_MOMENTUM,
        name=prefix + "project/BatchNorm",
    )(x)

    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + "Add")([shortcut, x])

    return x


@keras.utils.register_keras_serializable(package="keras_cv.models")
class MobileNetV3Backbone(Backbone):
    """Instantiates the MobileNetV3 architecture.

    References:
        - [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf)
        (ICCV 2019)
        - [Based on the Original keras.applications MobileNetv3](https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet_v3.py)

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        stackwise_expansion: list of ints or floats, the expansion ratio for
            each inverted residual block in the model.
        stackwise_filters: list of ints, number of filters for each inverted
            residual block in the model.
        stackwise_stride: list of ints, stride length for each inverted
            residual block in the model.
        filters: integer, the number of filters for the convolution layer.
        include_rescaling: bool, whether to rescale the inputs. If set to True,
            inputs will be passed through a `Rescaling(scale=1 / 255)`
            layer.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e., output of `layers.Input()`)
            to use as image input for the model.
        alpha: float, controls the width of the network. This is known as the
            depth multiplier in the MobileNetV3 paper, but the name is kept for
            consistency with MobileNetV1 in Keras.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                are used at each layer.
        dropout_rate: a float between 0 and 1 denoting the fraction of input
            units to drop during training, defaults to 0.2.

    Examples:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Randomly initialized backbone with a custom config
    model = MobileNetV3Backbone(
        stackwise_expansion=[1, 72.0 / 16, 88.0 / 24, 4, 6, 6, 3, 3, 6, 6, 6],
        stackwise_filters=[16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96],
        stackwise_stride=[2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1],
        filters=1024,
        include_rescaling=False,
    )
    output = model(input_data)
    ```
    """  # noqa: E501

    def __init__(
        self,
        *,
        stackwise_expansion,
        stackwise_filters,
        stackwise_kernel_size,
        stackwise_stride,
        stackwise_se_ratio,
        stackwise_activation,
        filters,
        include_rescaling,
        input_shape=(None, None, 3),
        input_tensor=None,
        alpha=1.0,
        dropout_rate=0.2,
        **kwargs,
    ):
        inputs = utils.parse_model_inputs(input_shape, input_tensor)
        x = inputs

        if include_rescaling:
            x = layers.Rescaling(scale=1 / 255)(x)

        x = layers.Conv2D(
            16,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            use_bias=False,
            name="Conv",
        )(x)
        x = layers.BatchNormalization(
            axis=CHANNEL_AXIS,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name="Conv/BatchNorm",
        )(x)
        x = apply_hard_swish(x)

        pyramid_level_inputs = {}
        for stack_index in range(len(stackwise_filters)):
            x = apply_inverted_res_block(
                x,
                expansion=stackwise_expansion[stack_index],
                filters=adjust_channels(
                    (stackwise_filters[stack_index]) * alpha
                ),
                kernel_size=stackwise_kernel_size[stack_index],
                stride=stackwise_stride[stack_index],
                se_ratio=stackwise_se_ratio[stack_index],
                activation=stackwise_activation[stack_index],
                expansion_index=stack_index,
            )
            pyramid_level_inputs[stack_index + 2] = x.node.layer.name

        last_conv_ch = adjust_channels(backend.int_shape(x)[CHANNEL_AXIS] * 6)

        # if the width multiplier is greater than 1 we
        # increase the number of output channels
        if alpha > 1.0:
            filters = adjust_channels(filters * alpha)
        x = layers.Conv2D(
            last_conv_ch,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name="Conv_1",
        )(x)
        x = layers.BatchNormalization(
            axis=CHANNEL_AXIS,
            epsilon=BN_EPSILON,
            momentum=BN_MOMENTUM,
            name="Conv_1/BatchNorm",
        )(x)
        x = apply_hard_swish(x)

        super().__init__(inputs=inputs, outputs=x, **kwargs)

        self.pyramid_level_inputs = pyramid_level_inputs
        self.stackwise_expansion = stackwise_expansion
        self.stackwise_filters = stackwise_filters
        self.stackwise_kernel_size = stackwise_kernel_size
        self.stackwise_stride = stackwise_stride
        self.stackwise_se_ratio = stackwise_se_ratio
        self.stackwise_activation = stackwise_activation
        self.filters = filters
        self.include_rescaling = include_rescaling
        self.input_tensor = input_tensor
        self.alpha = alpha
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stackwise_expansion": self.stackwise_expansion,
                "stackwise_filters": self.stackwise_filters,
                "stackwise_kernel_size": self.stackwise_kernel_size,
                "stackwise_stride": self.stackwise_stride,
                "stackwise_se_ratio": self.stackwise_se_ratio,
                "stackwise_activation": self.stackwise_activation,
                "filters": self.filters,
                "include_rescaling": self.include_rescaling,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "alpha": self.alpha,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy(backbone_presets)


ALIAS_DOCSTRING = """MobileNetV3Backbone model with {num_layers} layers.

    References:
        - [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
        - [Based on the Original keras.applications MobileNetv3](https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet_v3.py)

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        include_rescaling: bool, whether to rescale the inputs. If set to
            True, inputs will be passed through a `Rescaling(scale=1 / 255)`
            layer. Defaults to True.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e., output of `layers.Input()`)
            to use as image input for the model.

    Examples:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Randomly initialized backbone
    model = {name}Backbone()
    output = model(input_data)
    ```
"""  # noqa: E501


class MobileNetV3SmallBackbone(MobileNetV3Backbone):
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
        return MobileNetV3Backbone.from_preset("mobilenetv3small", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}


class MobileNetV3LargeBackbone(MobileNetV3Backbone):
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
        return MobileNetV3Backbone.from_preset("mobilenetv3large", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}


setattr(
    MobileNetV3LargeBackbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="MobileNetV3Large", num_layers="28"),
)
setattr(
    MobileNetV3SmallBackbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="MobileNetV3Small", num_layers="14"),
)
