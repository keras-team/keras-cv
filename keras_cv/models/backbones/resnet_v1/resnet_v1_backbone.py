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
"""ResNet models for KerasCV.
Reference:
  - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
    (CVPR 2015)
  - [Based on the original keras.applications ResNet](https://github.com/keras-team/keras/blob/master/keras/applications/resnet.py)  # noqa: E501
"""

import copy

from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers

from keras_cv.models import utils
from keras_cv.models.backbones.backbone import Backbone
from keras_cv.models.backbones.resnet_v1.resnet_v1_backbone_presets import (
    backbone_presets,
)
from keras_cv.models.backbones.resnet_v1.resnet_v1_backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.utils.python_utils import classproperty

BN_AXIS = 3
BN_EPSILON = 1.001e-5


def apply_basic_block(
    x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None
):
    """A basic residual block (v1).

    Args:
        x: input tensor.
        filters: int, filters of the basic layer.
        kernel_size: int, kernel size of the bottleneck layer, defaults to 3.
        stride: int, stride of the first layer, defaults to 1.
        conv_shortcut: bool, uses convolution shortcut if `True`. If `False`
            (default), uses identity or pooling shortcut, based on stride.
        name: string, optional prefix for the layer names used in the block.

    Returns:
      Output tensor for the residual block.
    """

    if name is None:
        name = f"v1_basic_block_{backend.get_uid('v1_basic_block_')}"

    if conv_shortcut:
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=stride,
            use_bias=False,
            name=name + "_0_conv",
        )(x)
        shortcut = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_0_bn"
        )(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="SAME",
        strides=stride,
        use_bias=False,
        name=name + "_1_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_1_bn"
    )(x)
    x = layers.Activation("relu", name=name + "_1_relu")(x)

    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="SAME",
        use_bias=False,
        name=name + "_2_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_2_bn"
    )(x)

    x = layers.Add(name=name + "_add")([shortcut, x])
    x = layers.Activation("relu", name=name + "_out")(x)
    return x


def apply_block(
    x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None
):
    """A residual block (v1).

    Args:
        x: input tensor.
        filters: int, filters of the basic layer.
        kernel_size: int, kernel size of the bottleneck layer, defaults to 3.
        stride: int, stride of the first layer, defaults to 1.
        conv_shortcut: bool, uses convolution shortcut if `True`. If `False`
            (default), uses identity or pooling shortcut, based on stride.
        name: string, optional prefix for the layer names used in the block.

    Returns:
      Output tensor for the residual block.
    """

    if name is None:
        name = f"v1_block_{backend.get_uid('v1_block')}"

    if conv_shortcut:
        shortcut = layers.Conv2D(
            4 * filters,
            1,
            strides=stride,
            use_bias=False,
            name=name + "_0_conv",
        )(x)
        shortcut = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_0_bn"
        )(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(
        filters, 1, strides=stride, use_bias=False, name=name + "_1_conv"
    )(x)
    x = layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_1_bn"
    )(x)
    x = layers.Activation("relu", name=name + "_1_relu")(x)

    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="SAME",
        use_bias=False,
        name=name + "_2_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_2_bn"
    )(x)
    x = layers.Activation("relu", name=name + "_2_relu")(x)

    x = layers.Conv2D(4 * filters, 1, use_bias=False, name=name + "_3_conv")(x)
    x = layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_3_bn"
    )(x)

    x = layers.Add(name=name + "_add")([shortcut, x])
    x = layers.Activation("relu", name=name + "_out")(x)
    return x


def apply_stack(
    x,
    filters,
    blocks,
    stride=2,
    name=None,
    block_type="block",
    first_shortcut=True,
):
    """A set of stacked residual blocks.

    Args:
        x: input tensor.
        filters: int, filters of the layer in a block.
        blocks: int, blocks in the stacked blocks.
        stride: int, stride of the first layer in the first block, defaults to
            2.
        name: string, optional prefix for the layer names used in the block.
        block_type: string, one of "basic_block" or "block". The block type to
              stack. Use "basic_block" for ResNet18 and ResNet34.
        first_shortcut: bool. Use convolution shortcut if `True` (default),
              otherwise uses identity or pooling shortcut, based on stride.

    Returns:
        Output tensor for the stacked blocks.
    """

    if name is None:
        name = "v1_stack"

    if block_type == "basic_block":
        block_fn = apply_basic_block
    elif block_type == "block":
        block_fn = apply_block
    else:
        raise ValueError(
            """`block_type` must be either "basic_block" or "block". """
            f"Received block_type={block_type}."
        )

    x = block_fn(
        x,
        filters,
        stride=stride,
        name=name + "_block1",
        conv_shortcut=first_shortcut,
    )
    for i in range(2, blocks + 1):
        x = block_fn(
            x, filters, conv_shortcut=False, name=name + "_block" + str(i)
        )
    return x


@keras.utils.register_keras_serializable(package="keras_cv.models")
class ResNetBackbone(Backbone):
    """Instantiates the ResNet architecture.

    Reference:
        - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

    The difference in ResNetV1 and ResNetV2 rests in the structure of their
    individual building blocks. In ResNetV2, the batch normalization and
    ReLU activation precede the convolution layers, as opposed to ResNetV1 where
    the batch normalization and ReLU activation are applied after the
    convolution layers.

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        stackwise_filters: list of ints, number of filters for each stack in
            the model.
        stackwise_blocks: list of ints, number of blocks for each stack in the
            model.
        stackwise_strides: list of ints, stride for each stack in the model.
        include_rescaling: bool, whether to rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        block_type: string, one of "basic_block" or "block". The block type to
            stack. Use "basic_block" for ResNet18 and ResNet34.

    Examples:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Pretrained backbone
    model = keras_cv.models.ResNetBackbone.from_preset("resnet50_imagenet")
    output = model(input_data)

    # Randomly initialized backbone with a custom config
    model = ResNetBackbone(
        stackwise_filters=[64, 128, 256, 512],
        stackwise_blocks=[2, 2, 2, 2],
        stackwise_strides=[1, 2, 2, 2],
        include_rescaling=False,
    )
    output = model(input_data)
    ```
    """  # noqa: E501

    def __init__(
        self,
        *,
        stackwise_filters,
        stackwise_blocks,
        stackwise_strides,
        include_rescaling,
        input_shape=(None, None, 3),
        input_tensor=None,
        block_type="block",
        **kwargs,
    ):
        inputs = utils.parse_model_inputs(input_shape, input_tensor)
        x = inputs

        if include_rescaling:
            x = layers.Rescaling(1 / 255.0)(x)

        x = layers.Conv2D(
            64, 7, strides=2, use_bias=False, padding="same", name="conv1_conv"
        )(x)

        x = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name="conv1_bn"
        )(x)
        x = layers.Activation("relu", name="conv1_relu")(x)

        x = layers.MaxPooling2D(
            3, strides=2, padding="same", name="pool1_pool"
        )(x)

        num_stacks = len(stackwise_filters)

        pyramid_level_inputs = {}
        for stack_index in range(num_stacks):
            x = apply_stack(
                x,
                filters=stackwise_filters[stack_index],
                blocks=stackwise_blocks[stack_index],
                stride=stackwise_strides[stack_index],
                block_type=block_type,
                first_shortcut=(block_type == "block" or stack_index > 0),
                name=f"v2_stack_{stack_index}",
            )
            pyramid_level_inputs[stack_index + 2] = x.node.layer.name

        # Create model.
        super().__init__(inputs=inputs, outputs=x, **kwargs)

        # All references to `self` below this line
        self.pyramid_level_inputs = pyramid_level_inputs
        self.stackwise_filters = stackwise_filters
        self.stackwise_blocks = stackwise_blocks
        self.stackwise_strides = stackwise_strides
        self.include_rescaling = include_rescaling
        self.input_tensor = input_tensor
        self.block_type = block_type

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stackwise_filters": self.stackwise_filters,
                "stackwise_blocks": self.stackwise_blocks,
                "stackwise_strides": self.stackwise_strides,
                "include_rescaling": self.include_rescaling,
                # Remove batch dimension from `input_shape`
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "block_type": self.block_type,
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


ALIAS_DOCSTRING = """ResNetBackbone (V1) model with {num_layers} layers.

    Reference:
        - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

    The difference in ResNetV1 and ResNetV2 rests in the structure of their
    individual building blocks. In ResNetV2, the batch normalization and
    ReLU activation precede the convolution layers, as opposed to ResNetV1 where
    the batch normalization and ReLU activation are applied after the
    convolution layers.

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        include_rescaling: bool, whether to rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    Examples:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Randomly initialized backbone
    model = ResNet{num_layers}Backbone()
    output = model(input_data)
    ```
"""  # noqa: E501


class ResNet18Backbone(ResNetBackbone):
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
        return ResNetBackbone.from_preset("resnet18", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class ResNet34Backbone(ResNetBackbone):
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
        return ResNetBackbone.from_preset("resnet34", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class ResNet50Backbone(ResNetBackbone):
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
        return ResNetBackbone.from_preset("resnet50", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "resnet50_imagenet": copy.deepcopy(
                backbone_presets["resnet50_imagenet"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


class ResNet101Backbone(ResNetBackbone):
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
        return ResNetBackbone.from_preset("resnet101", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class ResNet152Backbone(ResNetBackbone):
    def __new__(
        self,
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
        return ResNetBackbone.from_preset("resnet152", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


setattr(ResNet18Backbone, "__doc__", ALIAS_DOCSTRING.format(num_layers=18))
setattr(ResNet34Backbone, "__doc__", ALIAS_DOCSTRING.format(num_layers=34))
setattr(ResNet50Backbone, "__doc__", ALIAS_DOCSTRING.format(num_layers=50))
setattr(ResNet101Backbone, "__doc__", ALIAS_DOCSTRING.format(num_layers=101))
setattr(ResNet152Backbone, "__doc__", ALIAS_DOCSTRING.format(num_layers=152))
