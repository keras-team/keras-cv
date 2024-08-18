# Copyright 2022 The KerasCV Authors
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
"""ResNet backbone model.
Reference:
  - [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (ECCV 2016)
  - [Based on the original keras.applications ResNet](https://github.com/keras-team/keras/blob/master/keras/applications/resnet_v2.py)
"""  # noqa: E501

import copy

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.models import utils
from keras_cv.src.models.backbones.backbone import Backbone
from keras_cv.src.models.backbones.resnet_v2.resnet_v2_backbone_presets import (
    backbone_presets,
)
from keras_cv.src.models.backbones.resnet_v2.resnet_v2_backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.src.utils.python_utils import classproperty

BN_AXIS = 3
BN_EPSILON = 1.001e-5


@keras_cv_export("keras_cv.models.ResNetV2Backbone")
class ResNetV2Backbone(Backbone):
    """Instantiates the ResNetV2 architecture.

    Reference:
        - [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (ECCV 2016)

    The difference in Resnet and ResNetV2 rests in the structure of their
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
        stackwise_dilations: list of ints, dilation for each stack in the
            model. If `None` (default), dilation will not be used.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        block_type: string, one of "basic_block" or "block". The block type to
            stack. Use "basic_block" for smaller models like ResNet18 and
            ResNet34.

    Examples:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Pretrained backbone
    model = keras_cv.models.ResNetV2Backbone.from_preset("resnet50_v2_imagenet")
    output = model(input_data)

    # Randomly initialized backbone with a custom config
    model = ResNetV2Backbone(
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
        stackwise_dilations=None,
        input_shape=(None, None, 3),
        input_tensor=None,
        block_type="block",
        **kwargs,
    ):
        inputs = utils.parse_model_inputs(input_shape, input_tensor)
        x = inputs

        if include_rescaling:
            x = keras.layers.Rescaling(1 / 255.0)(x)

        x = keras.layers.Conv2D(
            64,
            7,
            strides=2,
            use_bias=True,
            padding="same",
            name="conv1_conv",
        )(x)

        x = keras.layers.MaxPooling2D(
            3, strides=2, padding="same", name="pool1_pool"
        )(x)

        num_stacks = len(stackwise_filters)
        if stackwise_dilations is None:
            stackwise_dilations = [1] * num_stacks

        pyramid_level_inputs = {}
        for stack_index in range(num_stacks):
            x = apply_stack(
                x,
                filters=stackwise_filters[stack_index],
                blocks=stackwise_blocks[stack_index],
                stride=stackwise_strides[stack_index],
                dilations=stackwise_dilations[stack_index],
                block_type=block_type,
                first_shortcut=(block_type == "block" or stack_index > 0),
                name=f"v2_stack_{stack_index}",
            )
            pyramid_level_inputs[f"P{stack_index + 2}"] = (
                utils.get_tensor_input_name(x)
            )

        x = keras.layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name="post_bn"
        )(x)
        x = keras.layers.Activation("relu", name="post_relu")(x)

        # Create model.
        super().__init__(inputs=inputs, outputs=x, **kwargs)

        # All references to `self` below this line
        self.pyramid_level_inputs = pyramid_level_inputs
        self.stackwise_filters = stackwise_filters
        self.stackwise_blocks = stackwise_blocks
        self.stackwise_strides = stackwise_strides
        self.include_rescaling = include_rescaling
        self.stackwise_dilations = stackwise_dilations
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
                "stackwise_dilations": self.stackwise_dilations,
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


def apply_basic_block(
    x,
    filters,
    kernel_size=3,
    stride=1,
    dilation=1,
    conv_shortcut=False,
    name=None,
):
    """A basic residual block (v2).

    Args:
        x: input tensor.
        filters: int, filters of the basic layer.
        kernel_size: int, kernel size of the bottleneck layer, defaults to 3.
        stride: int, stride of the first layer, defaults to 1.
        dilation: int, the dilation rate to use for dilated convolution.
            Defaults to 1.
        conv_shortcut: bool, uses convolution shortcut if `True`. If `False`
            (default), uses identity or pooling shortcut, based on stride.
        name: string, optional prefix for the layer names used in the block.

    Returns:
      Output tensor for the residual block.
    """

    if name is None:
        name = f"v2_basic_block_{keras.backend.get_uid('v2_basic_block')}"

    use_preactivation = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_use_preactivation_bn"
    )(x)

    use_preactivation = keras.layers.Activation(
        "relu", name=name + "_use_preactivation_relu"
    )(use_preactivation)

    s = stride if dilation == 1 else 1
    if conv_shortcut:
        shortcut = keras.layers.Conv2D(
            filters, 1, strides=s, name=name + "_0_conv"
        )(use_preactivation)
    else:
        shortcut = (
            keras.layers.MaxPooling2D(
                1, strides=stride, name=name + "_0_max_pooling"
            )(x)
            if s > 1
            else x
        )

    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        padding="SAME",
        strides=1,
        use_bias=False,
        name=name + "_1_conv",
    )(use_preactivation)
    x = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_1_bn"
    )(x)
    x = keras.layers.Activation("relu", name=name + "_1_relu")(x)

    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=s,
        padding="same",
        dilation_rate=dilation,
        use_bias=False,
        name=name + "_2_conv",
    )(x)

    x = keras.layers.Add(name=name + "_out")([shortcut, x])
    return x


def apply_block(
    x,
    filters,
    kernel_size=3,
    stride=1,
    dilation=1,
    conv_shortcut=False,
    name=None,
):
    """A residual block (v2).

    Args:
        x: input tensor.
        filters: int, filters of the basic layer.
        kernel_size: int, kernel size of the bottleneck layer, defaults to 3.
        stride: int, stride of the first layer, defaults to 1.
        dilation: int, the dilation rate to use for dilated convolution.
            Defaults to 1.
        conv_shortcut: bool, uses convolution shortcut if `True`. If `False`
            (default), uses identity or pooling shortcut, based on stride.
        name: string, optional prefix for the layer names used in the block.

    Returns:
      Output tensor for the residual block.
    """
    if name is None:
        name = f"v2_block_{keras.backend.get_uid('v2_block')}"

    use_preactivation = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_use_preactivation_bn"
    )(x)

    use_preactivation = keras.layers.Activation(
        "relu", name=name + "_use_preactivation_relu"
    )(use_preactivation)

    s = stride if dilation == 1 else 1
    if conv_shortcut:
        shortcut = keras.layers.Conv2D(
            4 * filters,
            1,
            strides=s,
            name=name + "_0_conv",
        )(use_preactivation)
    else:
        shortcut = (
            keras.layers.MaxPooling2D(
                1, strides=stride, name=name + "_0_max_pooling"
            )(x)
            if s > 1
            else x
        )

    x = keras.layers.Conv2D(
        filters, 1, strides=1, use_bias=False, name=name + "_1_conv"
    )(use_preactivation)
    x = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_1_bn"
    )(x)
    x = keras.layers.Activation("relu", name=name + "_1_relu")(x)

    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=s,
        use_bias=False,
        padding="same",
        dilation_rate=dilation,
        name=name + "_2_conv",
    )(x)
    x = keras.layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_2_bn"
    )(x)
    x = keras.layers.Activation("relu", name=name + "_2_relu")(x)

    x = keras.layers.Conv2D(4 * filters, 1, name=name + "_3_conv")(x)
    x = keras.layers.Add(name=name + "_out")([shortcut, x])
    return x


def apply_stack(
    x,
    filters,
    blocks,
    stride=2,
    dilations=1,
    name=None,
    block_type="block",
    first_shortcut=True,
):
    """A set of stacked blocks.

    Args:
        x: input tensor.
        filters: int, filters of the layer in a block.
        blocks: int, blocks in the stacked blocks.
        stride: int, stride of the first layer in the first block, defaults
            to 2.
        dilations: int, the dilation rate to use for dilated convolution.
            Defaults to 1.
        name: string, optional prefix for the layer names used in the block.
        block_type: string, one of "basic_block" or "block". The block type to
            stack. Use "basic_block" for ResNet18 and ResNet34.
        first_shortcut: bool. Use convolution shortcut if `True` (default),
            otherwise uses identity or pooling shortcut, based on stride.

    Returns:
        Output tensor for the stacked blocks.
    """

    if name is None:
        name = "v2_stack"

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
        x, filters, conv_shortcut=first_shortcut, name=name + "_block1"
    )
    for i in range(2, blocks):
        x = block_fn(
            x, filters, dilation=dilations, name=name + "_block" + str(i)
        )
    x = block_fn(
        x,
        filters,
        stride=stride,
        dilation=dilations,
        name=name + "_block" + str(blocks),
    )
    return x
