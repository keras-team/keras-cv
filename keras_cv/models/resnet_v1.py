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
"""ResNet models for KerasCV.
Reference:
  - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (CVPR 2015)
  - [Based on the original keras.applications ResNet](https://github.com/keras-team/keras/blob/master/keras/applications/resnet.py)
"""

import types

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers

from keras_cv.models import utils
from keras_cv.models.weights import parse_weights

MODEL_CONFIGS = {
    "ResNet18": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [2, 2, 2, 2],
        "stackwise_strides": [1, 2, 2, 2],
    },
    "ResNet34": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [3, 4, 6, 3],
        "stackwise_strides": [1, 2, 2, 2],
    },
    "ResNet50": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [3, 4, 6, 3],
        "stackwise_strides": [1, 2, 2, 2],
    },
    "ResNet101": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [3, 4, 23, 3],
        "stackwise_strides": [1, 2, 2, 2],
    },
    "ResNet152": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [3, 8, 36, 3],
        "stackwise_strides": [1, 2, 2, 2],
    },
}

BN_AXIS = 3
BN_EPSILON = 1.001e-5

BASE_DOCSTRING = """Instantiates the {name} architecture.
    Reference:
        - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
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
            the top of the network.  If provided, `num_classes` must be provided.
        num_classes: optional int, number of num_classes to classify images into (only
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
    x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None
):
    """A basic residual block (v1).

    Args:
        x: input tensor.
        filters: int, filters of the basic layer.
        kernel_size: int, kernel size of the bottleneck layer. Defaults to 3.
        stride: int, stride of the first layer. Defaults to 1.
        conv_shortcut: bool, uses convolution shortcut if `True`. If `False`
            (default), uses identity or pooling shortcut, based on stride.

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
        kernel_size: int, kernel size of the bottleneck layer. Defaults to 3.
        stride: int, stride of the first layer. Defaults to 1.
        conv_shortcut: bool, uses convolution shortcut if `True`. If `False`
            (default), uses identity or pooling shortcut, based on stride.

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
      stride: int, stride of the first layer in the first block. Defaults to 2.
      block_type: string, one of "basic_block" or "block". The block type to
            stack. Use "basic_block" for ResNet18 and ResNet34.
      first_shortcut: bool. Use convolution shortcut if `True` (default),
            otherwise uses identity or pooling shortcut, based on stride.

    Returns:
      Output tensor for the stacked blocks.
    """

    if name is None:
        name = f"v1_stack_{backend.get_uid('v1_stack')}"

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
class ResNet(keras.Model):
    """Instantiates the ResNet architecture.

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
        num_classes: optional number of num_classes to classify images
            into, only to be specified if `include_top` is True.
        classifier_activation: A `str` or callable. The activation function to
            use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top"
            layer.
        block_type: string, one of "basic_block" or "block". The block type to
            stack. Use "basic_block" for ResNet18 and ResNet34.

    Returns:
      A `keras.Model` instance.
    """

    def __init__(
        self,
        stackwise_filters,
        stackwise_blocks,
        stackwise_strides,
        include_rescaling,
        include_top,
        name="ResNet",
        weights=None,
        input_shape=(None, None, 3),
        input_tensor=None,
        pooling=None,
        num_classes=None,
        classifier_activation="softmax",
        block_type="block",
        **kwargs,
    ):
        if weights and not tf.io.gfile.exists(weights):
            raise ValueError(
                "The `weights` argument should be either `None` or the path to the "
                f"weights file to be loaded. Weights file not found at location: {weights}"
            )

        if include_top and not num_classes:
            raise ValueError(
                "If `include_top` is True, you should specify `num_classes`. "
                f"Received: num_classes={num_classes}"
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

        stack_level_outputs = {}
        for stack_index in range(num_stacks):
            x = apply_stack(
                x,
                filters=stackwise_filters[stack_index],
                blocks=stackwise_blocks[stack_index],
                stride=stackwise_strides[stack_index],
                block_type=block_type,
                first_shortcut=(block_type == "block" or stack_index > 0),
            )
            stack_level_outputs[stack_index + 2] = x

        if include_top:
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            x = layers.Dense(
                num_classes, activation=classifier_activation, name="predictions"
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
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.num_classes = num_classes
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
            "input_tensor": self.input_tensor,
            "pooling": self.pooling,
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
            "block_type": self.block_type,
            "name": self.name,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def ResNet18(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="resnet18",
    **kwargs,
):
    """Instantiates the ResNet18 architecture."""

    return ResNet(
        stackwise_filters=MODEL_CONFIGS["ResNet18"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["ResNet18"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["ResNet18"]["stackwise_strides"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        block_type="basic_block",
        **kwargs,
    )


def ResNet34(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="resnet34",
    **kwargs,
):
    """Instantiates the ResNet34 architecture."""

    return ResNet(
        stackwise_filters=MODEL_CONFIGS["ResNet34"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["ResNet34"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["ResNet34"]["stackwise_strides"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        block_type="basic_block",
        **kwargs,
    )


def ResNet50(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="resnet50",
    **kwargs,
):
    """Instantiates the ResNet50 architecture."""

    return ResNet(
        stackwise_filters=MODEL_CONFIGS["ResNet50"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["ResNet50"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["ResNet50"]["stackwise_strides"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        name=name,
        weights=parse_weights(weights, include_top, "resnet50"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        block_type="block",
        **kwargs,
    )


def ResNet101(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="resnet101",
    **kwargs,
):
    """Instantiates the ResNet101 architecture."""
    return ResNet(
        stackwise_filters=MODEL_CONFIGS["ResNet101"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["ResNet101"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["ResNet101"]["stackwise_strides"],
        name=name,
        include_rescaling=include_rescaling,
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        block_type="block",
        **kwargs,
    )


def ResNet152(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="resnet152",
    **kwargs,
):
    """Instantiates the ResNet152 architecture."""
    return ResNet(
        stackwise_filters=MODEL_CONFIGS["ResNet152"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["ResNet152"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["ResNet152"]["stackwise_strides"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        block_type="block",
        **kwargs,
    )


setattr(ResNet18, "__doc__", BASE_DOCSTRING.format(name="ResNet18"))
setattr(ResNet34, "__doc__", BASE_DOCSTRING.format(name="ResNet34"))
setattr(ResNet50, "__doc__", BASE_DOCSTRING.format(name="ResNet50"))
setattr(ResNet101, "__doc__", BASE_DOCSTRING.format(name="ResNet101"))
setattr(ResNet152, "__doc__", BASE_DOCSTRING.format(name="ResNet152"))
