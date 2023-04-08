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

"""MobileNet v3 backbone model for KerasCV.

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

channel_axis = -1


def adjust_channels(x, divisor=8, min_value=None):
    """Ensure that all layers have a channel number that is divisible by the
    `divisor`.

    Args:
        x: integer, input value.
        divisor: integer, the value by which a channel number should be
            divisible, defaults to 8.
        min_value: float, minimum value for the new tensor.

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


def apply_hard_sigmoid(x, name=None):
    """The Hard Sigmoid function.

    Args:
        x: input tensor
        name: string, layer label.

    Returns:
        the updated input tensor.
    """
    if name is None:
        name = f"hard_sigmoid_{backend.get_uid('hard_sigmoid')}"

    activation = layers.ReLU(6.0)

    return activation(x + 3.0) * (1.0 / 6.0)


def apply_hard_swish(x, name=None):
    """The Hard Swish function.

    Args:
        x: input tensor
        name: string, layer label.

    Returns:
        the updated input tensor.
    """
    if name is None:
        name = f"hard_swish_{backend.get_uid('hard_swish')}"

    multiply_layer = layers.Multiply()

    return multiply_layer([x, apply_hard_sigmoid(x)])


def apply_inverted_res_block(
    x,
    expansion,
    filters,
    kernel_size,
    stride,
    se_ratio,
    activation,
    expansion_index,
    name=None,
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
            expanded convolutions.
        name: string, layer label.

    Returns:
        the updated input tensor.
    """
    if name is None:
        name = f"inverted_res_block_{backend.get_uid('inverted_res_block')}"

    shortcut = x
    prefix = "expanded_conv/"
    infilters = backend.int_shape(x)[channel_axis]

    if expansion_index:
        prefix = f"expanded_conv_{expansion_index}"

        x = layers.Conv2D(
            adjust_channels(infilters * expansion),
            kernel_size=1,
            padding="same",
            use_bias=False,
            name=prefix + "expand",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
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
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
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
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "project/BatchNorm",
    )(x)

    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + "Add")([shortcut, x])

    return x


def stack_fn_v3small(x, kernel, activation, se_ratio, alpha=1.0):
    x = apply_inverted_res_block(
        x, 1, adjust_channels(16 * alpha), 3, 2, se_ratio, layers.ReLU(), 0
    )
    x = apply_inverted_res_block(
        x, 72.0 / 16, adjust_channels(24 * alpha), 3, 2, None, layers.ReLU(), 1
    )
    x = apply_inverted_res_block(
        x, 88.0 / 24, adjust_channels(24 * alpha), 3, 1, None, layers.ReLU(), 2
    )
    x = apply_inverted_res_block(
        x, 4, adjust_channels(40 * alpha), kernel, 2, se_ratio, activation, 3
    )
    x = apply_inverted_res_block(
        x, 6, adjust_channels(40 * alpha), kernel, 1, se_ratio, activation, 4
    )
    x = apply_inverted_res_block(
        x, 6, adjust_channels(40 * alpha), kernel, 1, se_ratio, activation, 5
    )
    x = apply_inverted_res_block(
        x, 3, adjust_channels(48 * alpha), kernel, 1, se_ratio, activation, 6
    )
    x = apply_inverted_res_block(
        x, 3, adjust_channels(48 * alpha), kernel, 1, se_ratio, activation, 7
    )
    x = apply_inverted_res_block(
        x, 6, adjust_channels(96 * alpha), kernel, 2, se_ratio, activation, 8
    )
    x = apply_inverted_res_block(
        x, 6, adjust_channels(96 * alpha), kernel, 1, se_ratio, activation, 9
    )
    x = apply_inverted_res_block(
        x, 6, adjust_channels(96 * alpha), kernel, 1, se_ratio, activation, 10
    )
    return x


def stack_fn_v3large(x, kernel, activation, se_ratio, alpha=1.0):
    x = apply_inverted_res_block(
        x, 1, adjust_channels(16 * alpha), 3, 1, None, layers.ReLU(), 0
    )
    x = apply_inverted_res_block(
        x, 4, adjust_channels(24 * alpha), 3, 2, None, layers.ReLU(), 1
    )
    x = apply_inverted_res_block(
        x, 3, adjust_channels(24 * alpha), 3, 1, None, layers.ReLU(), 2
    )
    x = apply_inverted_res_block(
        x, 3, adjust_channels(40 * alpha), kernel, 2, se_ratio, layers.ReLU(), 3
    )
    x = apply_inverted_res_block(
        x, 3, adjust_channels(40 * alpha), kernel, 1, se_ratio, layers.ReLU(), 4
    )
    x = apply_inverted_res_block(
        x, 3, adjust_channels(40 * alpha), kernel, 1, se_ratio, layers.ReLU(), 5
    )
    x = apply_inverted_res_block(
        x, 6, adjust_channels(80 * alpha), 3, 2, None, activation, 6
    )
    x = apply_inverted_res_block(
        x, 2.5, adjust_channels(80 * alpha), 3, 1, None, activation, 7
    )
    x = apply_inverted_res_block(
        x, 2.3, adjust_channels(80 * alpha), 3, 1, None, activation, 8
    )
    x = apply_inverted_res_block(
        x, 2.3, adjust_channels(80 * alpha), 3, 1, None, activation, 9
    )
    x = apply_inverted_res_block(
        x, 6, adjust_channels(112 * alpha), 3, 1, se_ratio, activation, 10
    )
    x = apply_inverted_res_block(
        x, 6, adjust_channels(112 * alpha), 3, 1, se_ratio, activation, 11
    )
    x = apply_inverted_res_block(
        x, 6, adjust_channels(160 * alpha), kernel, 2, se_ratio, activation, 12
    )
    x = apply_inverted_res_block(
        x, 6, adjust_channels(160 * alpha), kernel, 1, se_ratio, activation, 13
    )
    x = apply_inverted_res_block(
        x, 6, adjust_channels(160 * alpha), kernel, 1, se_ratio, activation, 14
    )
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
        stack_fn: a function that returns tensors passed through Inverted
            Residual Blocks.
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
        minimalistic: in addition to large and small models, this module also
            contains so-called minimalistic models; these models have the same
            per-layer dimensions characteristic as MobilenetV3 however, they
            don't utilize any of the advanced blocks (squeeze-and-excite units,
            hard-swish, and 5x5 convolutions). While these models are less
            efficient on CPU, they are much more performant on GPU/DSP.
        dropout_rate: a float between 0 and 1 denoting the fraction of input
            units to drop, defaults to 0.2.

    Examples:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Randomly initialized backbone with a custom config
    model = MobileNetV3Backbone(
        stack_fn=None,
        filters=1024,
        include_rescaling=False,
    )
    output = model(input_data)
    ```
    """  # noqa: E501

    def __init__(
        self,
        *,
        stack_fn_type="MobileNetV3Small",
        filters,
        include_rescaling,
        input_shape=(None, None, 3),
        input_tensor=None,
        alpha=1.0,
        minimalistic=True,
        dropout_rate=0.2,
        **kwargs,
    ):
        if stack_fn_type == "MobileNetV3Small":
            stack_fn = stack_fn_v3small
        elif stack_fn_type == "MobileNetV3Large":
            stack_fn = stack_fn_v3large
        else:
            raise ValueError(
                """`stack_fn_type` must be either "MobileNetV3Small" or
                "MobileNetV3Large". """
                f"Received stack_fn_type={stack_fn_type}."
            )

        if minimalistic:
            kernel = 3
            activation = layers.ReLU()
            se_ratio = None
        else:
            kernel = 5
            activation = apply_hard_swish
            se_ratio = 0.25

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
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name="Conv/BatchNorm",
        )(x)
        x = activation(x)

        x = stack_fn(x, kernel, activation, se_ratio)

        last_conv_ch = adjust_channels(backend.int_shape(x)[channel_axis] * 6)

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
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name="Conv_1/BatchNorm",
        )(x)
        x = activation(x)

        super().__init__(inputs=inputs, outputs=x, **kwargs)

        self.stack_fn = stack_fn
        self.filters = filters
        self.include_rescaling = include_rescaling
        self.input_tensor = input_tensor
        self.alpha = alpha
        self.minimalistic = minimalistic
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stack_fn": self.stack_fn,
                "filters": self.filters,
                "include_rescaling": self.include_rescaling,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "alpha": self.alpha,
                "minimalistic": self.minimalistic,
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
        include_rescaling: bool, whether or not to rescale the inputs. If set to
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
        stack_fn_type="MobileNetV3Small",
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
        stack_fn_type="MobileNetV3Large",
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
