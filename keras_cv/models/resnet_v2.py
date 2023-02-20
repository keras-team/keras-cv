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
"""ResNet models for Keras.
Reference:
  - [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (ECCV 2016)
  - [Based on the original keras.applications ResNet](https://github.com/keras-team/keras/blob/master/keras/applications/resnet_v2.py)
"""

import types

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers

from keras_cv.models import utils
from keras_cv.models.weights import parse_weights

MODEL_CONFIGS = {
    "ResNet18V2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [2, 2, 2, 2],
        "stackwise_strides": [1, 2, 2, 2],
    },
    "ResNet34V2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [3, 4, 6, 3],
        "stackwise_strides": [1, 2, 2, 2],
    },
    "ResNet50V2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [3, 4, 6, 3],
        "stackwise_strides": [1, 2, 2, 2],
    },
    "ResNet101V2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [3, 4, 23, 3],
        "stackwise_strides": [1, 2, 2, 2],
    },
    "ResNet152V2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [3, 8, 36, 3],
        "stackwise_strides": [1, 2, 2, 2],
    },
}

BN_AXIS = 3
BN_EPSILON = 1.001e-5

BASE_DOCSTRING = """Instantiates the {name} architecture.
    Reference:
        - [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (ECCV 2016)
    This function returns a Keras {name} model.

    The difference in Resnet and ResNetV2 rests in the structure of their
    individual building blocks. In ResNetV2, the batch normalization and
    ReLU activation precede the convolution layers, as opposed to ResNetV1 where
    the batch normalization and ReLU activation are applied after the
    convolution layers.

    For transfer learning use cases, make sure to read the [guide to transfer
        learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        include_rescaling: bool, whether or not to Rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        include_top: bool, whether to include the fully-connected layer at
            the top of the network.  If provided, `classes` must be provided.
        classes: optional int, number of classes to classify images into (only
            to be specified if `include_top` is `True`).
        weights: one of `None` (random initialization), a pretrained weight file
            path, or a reference to pre-trained weights (e.g. 'imagenet/classification')
            (see available pre-trained weights in weights.py)
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be the 4D tensor output
                of the last convolutional block.
            - `avg` means that global average pooling will be applied to the output
                of the last convolutional block, and thus the output of the model will
                be a 2D tensor.
            - `max` means that global max pooling will be applied.
        name: (Optional) name to pass to the model.  Defaults to "{name}".
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
      A `keras.Model` instance.
"""


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
        kernel_size: int, kernel size of the bottleneck layer. Defaults to 3.
        stride: int, stride of the first layer. Defaults to 1.
        dilation: int, the dilation rate to use for dilated convolution.
            Defaults to 1.
        conv_shortcut: bool, uses convolution shortcut if `True`. If `False`
            (default), uses identity or pooling shortcut, based on stride.

    Returns:
      Output tensor for the residual block.
    """

    if name is None:
        name = f"v2_basic_block_{backend.get_uid('v2_basic_block')}"

    use_preactivation = layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_use_preactivation_bn"
    )(x)

    use_preactivation = layers.Activation(
        "relu", name=name + "_use_preactivation_relu"
    )(use_preactivation)

    s = stride if dilation == 1 else 1
    if conv_shortcut:
        shortcut = layers.Conv2D(filters, 1, strides=s, name=name + "_0_conv")(
            use_preactivation
        )
    else:
        shortcut = (
            layers.MaxPooling2D(
                1, strides=stride, name=name + "_0_max_pooling"
            )(x)
            if s > 1
            else x
        )

    x = layers.Conv2D(
        filters,
        kernel_size,
        padding="SAME",
        strides=1,
        use_bias=False,
        name=name + "_1_conv",
    )(use_preactivation)
    x = layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_1_bn"
    )(x)
    x = layers.Activation("relu", name=name + "_1_relu")(x)

    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=s,
        padding="same",
        dilation_rate=dilation,
        use_bias=False,
        name=name + "_2_conv",
    )(x)

    x = layers.Add(name=name + "_out")([shortcut, x])
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
        kernel_size: int, kernel size of the bottleneck layer. Defaults to 3.
        stride: int, stride of the first layer. Defaults to 1.
        dilation: int, the dilation rate to use for dilated convolution.
            Defaults to 1.
        conv_shortcut: bool, uses convolution shortcut if `True`. If `False`
            (default), uses identity or pooling shortcut, based on stride.

    Returns:
      Output tensor for the residual block.
    """
    if name is None:
        name = f"v2_block_{backend.get_uid('v2_block')}"

    use_preactivation = layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_use_preactivation_bn"
    )(x)

    use_preactivation = layers.Activation(
        "relu", name=name + "_use_preactivation_relu"
    )(use_preactivation)

    s = stride if dilation == 1 else 1
    if conv_shortcut:
        shortcut = layers.Conv2D(
            4 * filters,
            1,
            strides=s,
            name=name + "_0_conv",
        )(use_preactivation)
    else:
        shortcut = (
            layers.MaxPooling2D(
                1, strides=stride, name=name + "_0_max_pooling"
            )(x)
            if s > 1
            else x
        )

    x = layers.Conv2D(
        filters, 1, strides=1, use_bias=False, name=name + "_1_conv"
    )(use_preactivation)
    x = layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_1_bn"
    )(x)
    x = layers.Activation("relu", name=name + "_1_relu")(x)

    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=s,
        use_bias=False,
        padding="same",
        dilation_rate=dilation,
        name=name + "_2_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=name + "_2_bn"
    )(x)
    x = layers.Activation("relu", name=name + "_2_relu")(x)

    x = layers.Conv2D(4 * filters, 1, name=name + "_3_conv")(x)
    x = layers.Add(name=name + "_out")([shortcut, x])
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
    stack_index=1,
):
    """A set of stacked blocks.

    Args:
        x: input tensor.
        filters: int, filters of the layer in a block.
        blocks: int, blocks in the stacked blocks.
        stride: int, stride of the first layer in the first block. Defaults to 2.
        dilation: int, the dilation rate to use for dilated convolution.
            Defaults to 1.
        block_type: string, one of "basic_block" or "block". The block type to
            stack. Use "basic_block" for ResNet18 and ResNet34.
        first_shortcut: bool. Use convolution shortcut if `True` (default),
            otherwise uses identity or pooling shortcut, based on stride.

    Returns:
        Output tensor for the stacked blocks.
    """

    if name is None:
        name = f"v2_stack_{stack_index}"

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


@keras.utils.register_keras_serializable(package="keras_cv.models")
class ResNetV2(keras.Model):
    """Instantiates the ResNetV2 architecture.

    Args:
        stackwise_filters: list of ints, number of filters for each stack in
            the model.
        stackwise_blocks: list of ints, number of blocks for each stack in the
            model.
        stackwise_strides: list of ints, stride for each stack in the model.
        include_rescaling: bool, whether or not to Rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        include_top: bool, whether to include the fully-connected
            layer at the top of the network.
        stackwise_dialations: list of ints, dialation for each stack in the
            model. If `None` (default), dialation will not be used.
        name: string, model name.
        weights: one of `None` (random initialization),
            or the path to the weights file to be loaded.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True.
        classifier_activation: A `str` or callable. The activation function to
            use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top"
            layer.
        block_type: string, one of "basic_block" or "block". The block type to
            stack. Use "basic_block" for ResNet18 and ResNet34.
    """

    def __init__(
        self,
        stackwise_filters,
        stackwise_blocks,
        stackwise_strides,
        include_rescaling,
        include_top,
        stackwise_dilations=None,
        weights=None,
        input_shape=(None, None, 3),
        input_tensor=None,
        pooling=None,
        classes=None,
        classifier_activation="softmax",
        block_type="block",
        **kwargs,
    ):
        if weights and not tf.io.gfile.exists(weights):
            raise ValueError(
                "The `weights` argument should be either `None` or the path to the "
                "weights file to be loaded. Weights file not found at location: {weights}"
            )

        if include_top and not classes:
            raise ValueError(
                "If `include_top` is True, you should specify `classes`. "
                f"Received: classes={classes}"
            )

        if include_top and pooling:
            raise ValueError(
                f"`pooling` must be `None` when `include_top=True`."
                f"Received pooling={pooling} and include_top={include_top}. "
            )

        inputs = utils.parse_model_inputs(input_shape, input_tensor)
        x = inputs

        if include_rescaling:
            x = layers.Rescaling(1 / 255.0)(x)

        x = layers.Conv2D(
            64,
            7,
            strides=2,
            use_bias=True,
            padding="same",
            name="conv1_conv",
        )(x)

        x = layers.MaxPooling2D(
            3, strides=2, padding="same", name="pool1_pool"
        )(x)

        num_stacks = len(stackwise_filters)
        if stackwise_dilations is None:
            stackwise_dilations = [1] * num_stacks

        stack_level_outputs = {}
        for stack_index in range(num_stacks):
            x = apply_stack(
                x,
                filters=stackwise_filters[stack_index],
                blocks=stackwise_blocks[stack_index],
                stride=stackwise_strides[stack_index],
                dilations=stackwise_dilations[stack_index],
                block_type=block_type,
                first_shortcut=(block_type == "block" or stack_index > 0),
                stack_index=stack_index,
            )
            stack_level_outputs[stack_index + 2] = x

        x = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name="post_bn"
        )(x)
        x = layers.Activation("relu", name="post_relu")(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            x = layers.Dense(
                classes, activation=classifier_activation, name="predictions"
            )(x)
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D(name="max_pool")(x)

        # Create model.
        super().__init__(inputs=inputs, outputs=x, **kwargs)

        # All references to `self` below this line
        if weights is not None:
            self.load_weights(weights)
        # Set this private attribute for recreate backbone model with outputs at
        # each resolution level.
        self._backbone_level_outputs = stack_level_outputs

        # Bind the `to_backbone_model` method to the application model.
        self.as_backbone = types.MethodType(utils.as_backbone, self)

        self.stackwise_filters = stackwise_filters
        self.stackwise_blocks = stackwise_blocks
        self.stackwise_strides = stackwise_strides
        self.include_rescaling = include_rescaling
        self.include_top = include_top
        self.stackwise_dilations = stackwise_dilations
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.classes = classes
        self.classifier_activation = classifier_activation
        self.block_type = block_type

    def get_config(self):
        return {
            "stackwise_filters": self.stackwise_filters,
            "stackwise_blocks": self.stackwise_blocks,
            "stackwise_strides": self.stackwise_strides,
            "include_rescaling": self.include_rescaling,
            "include_top": self.include_top,
            # Remove batch dimension from `input_shape`
            "input_shape": self.input_shape[1:],
            "stackwise_dilations": self.stackwise_dilations,
            "input_tensor": self.input_tensor,
            "pooling": self.pooling,
            "classes": self.classes,
            "classifier_activation": self.classifier_activation,
            "block_type": self.block_type,
            "name": self.name,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def ResNet18V2(
    *,
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="resnet18",
    **kwargs,
):
    """Instantiates the ResNet18 architecture."""

    return ResNetV2(
        stackwise_filters=MODEL_CONFIGS["ResNet18V2"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["ResNet18V2"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["ResNet18V2"]["stackwise_strides"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        block_type="basic_block",
        **kwargs,
    )


def ResNet34V2(
    *,
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="resnet34",
    **kwargs,
):
    """Instantiates the ResNet34 architecture."""

    return ResNetV2(
        stackwise_filters=MODEL_CONFIGS["ResNet34V2"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["ResNet34V2"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["ResNet34V2"]["stackwise_strides"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        block_type="basic_block",
        **kwargs,
    )


def ResNet50V2(
    *,
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="resnet50v2",
    **kwargs,
):
    """Instantiates the ResNet50V2 architecture."""

    return ResNetV2(
        stackwise_filters=MODEL_CONFIGS["ResNet50V2"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["ResNet50V2"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["ResNet50V2"]["stackwise_strides"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        name=name,
        weights=parse_weights(weights, include_top, "resnet50v2"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        block_type="block",
        **kwargs,
    )


def ResNet101V2(
    *,
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="resnet101v2",
    **kwargs,
):
    """Instantiates the ResNet101V2 architecture."""
    return ResNetV2(
        stackwise_filters=MODEL_CONFIGS["ResNet101V2"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["ResNet101V2"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["ResNet101V2"]["stackwise_strides"],
        name=name,
        include_rescaling=include_rescaling,
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        block_type="block",
        **kwargs,
    )


def ResNet152V2(
    *,
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="resnet152v2",
    **kwargs,
):
    """Instantiates the ResNet152V2 architecture."""
    return ResNetV2(
        stackwise_filters=MODEL_CONFIGS["ResNet152V2"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["ResNet152V2"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["ResNet152V2"]["stackwise_strides"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        block_type="block",
        **kwargs,
    )


setattr(ResNet18V2, "__doc__", BASE_DOCSTRING.format(name="ResNet18V2"))
setattr(ResNet34V2, "__doc__", BASE_DOCSTRING.format(name="ResNet34V2"))
setattr(ResNet50V2, "__doc__", BASE_DOCSTRING.format(name="ResNet50V2"))
setattr(ResNet101V2, "__doc__", BASE_DOCSTRING.format(name="ResNet101V2"))
setattr(ResNet152V2, "__doc__", BASE_DOCSTRING.format(name="ResNet152V2"))
