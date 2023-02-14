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
        include_rescaling: whether or not to Rescale the inputs.If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        include_top: whether to include the fully-connected layer at the top of the
            network.  If provided, classes must be provided.
        classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True.
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


@keras.utils.register_keras_serializable(package="keras_cv.models.resnet_v2")
class BasicBlock(keras.layers.Layer):
    """A basic residual block (v2).

    Args:
        filters: integer, filters of the basic layer.
        kernel_size: integer, kernel size of the bottleneck layer. Defaults to 3.
        stride: integer, stride of the first layer. Defaults to 1.
        dilation: integer, the dilation rate to use for dilated convolution. Defaults to 1.
        conv_shortcut: bool, uses convolution shortcut if `True`. Defaults to `False`.
          otherwise identity shortcut.

    Returns:
      Output tensor for the residual block.
    """

    def __init__(
        self,
        filters,
        kernel_size=3,
        stride=1,
        dilation=1,
        conv_shortcut=False,
        **kwargs,
    ):
        if "name" not in kwargs:
            kwargs["name"] = f"v2_basic_block_{backend.get_uid('v2_basic_block')}"
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.stride2 = stride if dilation == 1 else 1
        self.conv_shortcut = conv_shortcut
        super().__init__(**kwargs)

        self.bn_preactivation = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name=self.name + "_use_preactivation_bn"
        )
        self.relu_preactivation = layers.Activation(
            "relu", name=self.name + "_use_preactivation_relu"
        )
        if conv_shortcut:
            self.shortcut = layers.Conv2D(
                self.filters, 1, strides=self.stride2, name=self.name + "_0_conv"
            )
        else:
            self.shortcut = (
                layers.MaxPooling2D(
                    1, strides=stride, name=self.name + "_0_max_pooling"
                )
                if self.stride2 > 1
                else None
            )
        self.conv1 = layers.Conv2D(
            self.filters,
            self.kernel_size,
            padding="SAME",
            strides=1,
            use_bias=False,
            name=self.name + "_1_conv",
        )
        self.bn1 = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name=self.name + "_1_bn"
        )
        self.activation1 = layers.Activation("relu", name=self.name + "_1_relu")
        self.conv2 = layers.Conv2D(
            self.filters,
            self.kernel_size,
            strides=self.stride2,
            padding="same",
            dilation_rate=self.dilation,
            use_bias=False,
            name=self.name + "_2_conv",
        )
        self.add = layers.Add(name=self.name + "_out")

    def call(self, inputs):
        x = self.bn_preactivation(inputs)
        x = self.relu_preactivation(x)
        shortcut = self.shortcut(x) if self.shortcut else x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        return self.add([x, shortcut])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "dilation": self.dilation,
                "conv_shortcut": self.conv_shortcut,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="keras_cv.models.resnet_v2")
class Block(keras.layers.Layer):
    """A residual block (v2).

    Args:
        filters: integer, filters of the basic layer.
        kernel_size: integer, kernel size of the bottleneck layer. Defaults to 3.
        stride: integer, stride of the first layer. Defaults to 1.
        dilation: integer, the dilation rate to use for dilated convolution. Defaults to 1.
        conv_shortcut: bool, uses convolution shortcut if `True`. Defaults to `False`.
          otherwise identity shortcut.

    Returns:
      Output tensor for the residual block.
    """

    def __init__(
        self,
        filters,
        kernel_size=3,
        stride=1,
        dilation=1,
        conv_shortcut=False,
        **kwargs,
    ):
        if "name" not in kwargs:
            kwargs["name"] = f"v2_block_{backend.get_uid('v2_block')}"
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.stride2 = stride if dilation == 1 else 1
        self.conv_shortcut = conv_shortcut
        super().__init__(**kwargs)

        self.bn_preactivation = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name=self.name + "_use_preactivation_bn"
        )
        self.relu_preactivation = layers.Activation(
            "relu", name=self.name + "_use_preactivation_relu"
        )
        if conv_shortcut:
            self.shortcut = layers.Conv2D(
                4 * filters,
                1,
                strides=self.stride2,
                name=self.name + "_0_conv",
            )
        else:
            self.shortcut = (
                layers.MaxPooling2D(
                    1, strides=stride, name=self.name + "_0_max_pooling"
                )
                if self.stride2 > 1
                else None
            )
        self.conv1 = layers.Conv2D(
            self.filters,
            kernel_size=1,
            strides=1,
            use_bias=False,
            name=self.name + "_1_conv",
        )
        self.bn1 = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name=self.name + "_1_bn"
        )
        self.activation1 = layers.Activation("relu", name=self.name + "_1_relu")
        self.conv2 = layers.Conv2D(
            self.filters,
            self.kernel_size,
            strides=self.stride2,
            padding="same",
            dilation_rate=self.dilation,
            use_bias=False,
            name=self.name + "_2_conv",
        )
        self.bn2 = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name=self.name + "_2_bn"
        )
        self.activation2 = layers.Activation("relu", name=self.name + "_2_relu")
        self.conv3 = layers.Conv2D(
            4 * filters, kernel_size=1, name=self.name + "_3_conv"
        )
        self.add = layers.Add(name=self.name + "_out")

    def call(self, inputs):
        x = self.bn_preactivation(inputs)
        x = self.relu_preactivation(x)
        shortcut = self.shortcut(x) if self.shortcut else x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.conv3(x)
        return self.add([x, shortcut])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "dilation": self.dilation,
                "conv_shortcut": self.conv_shortcut,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="keras_cv.models.resnet_v2")
class Stack(keras.layers.Layer):
    """A set of stacked blocks.

    Args:
        filters: integer, filters of the layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride: integer, stride of the first layer in the first block. Defaults to 2.
        dilation: integer, the dilation rate to use for dilated convolution. Defaults to 1.
        block_fn: callable, `Block` or `BasicBlock`, the block function to stack.
        first_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.

    Returns:
        Output tensor for the stacked blocks.
    """

    def __init__(
        self,
        filters,
        blocks,
        stride=2,
        dilation=1,
        block_fn=Block,
        first_shortcut=True,
        stack_index=1,
        **kwargs,
    ):
        if "name" not in kwargs:
            kwargs["name"] = f"v2_stack_{stack_index}"
        self.filters = filters
        self.blocks = blocks
        self.stride = stride
        self.dilation = dilation
        self.block_fn = block_fn
        self.first_shortcut = first_shortcut
        self.stack_index = stack_index
        super().__init__(**kwargs)

        self.first_block = block_fn(
            filters, conv_shortcut=first_shortcut, name=self.name + "_block1"
        )
        self.middle_blocks = [
            block_fn(filters, dilation=dilation, name=self.name + "_block" + str(i))
            for i in range(2, blocks)
        ]
        self.last_block = block_fn(
            filters,
            stride=stride,
            dilation=dilation,
            name=self.name + "_block" + str(blocks),
        )

    def call(self, inputs):
        x = self.first_block(inputs)
        for block in self.middle_blocks:
            x = block(x)
        return self.last_block(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "blocks": self.blocks,
                "stride": self.stride,
                "dilation": self.dilation,
                "block_fn": keras.utils.get_registered_name(self.block_fn),
                "first_shortcut": self.first_shortcut,
                "stack_index": self.stack_index,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        block_class_name = config["block_fn"]
        config["block_fn"] = keras.utils.get_registered_object(block_class_name)
        return super().from_config(config)


@keras.utils.register_keras_serializable(package="keras_cv.models")
class ResNetV2(keras.Model):
    """Instantiates the ResNetV2 architecture.

    Args:
        stackwise_filters: int, number of filters for each stack in the model.
        stackwise_blocks: int, number of blocks for each stack in the model.
        stackwise_strides: int, stride for each stack in the model.
        include_rescaling: bool, whether or not to Rescale the inputs. If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        include_top: bool, whether to include the fully-connected
            layer at the top of the network.
        weights: optional string, one of `None` (random initialization),
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
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.
        block_fn: callable, `Block` or `BasicBlock`, the block function to stack.
            Use 'basic_block' for ResNet18 and ResNet34.
        **kwargs: Pass-through keyword arguments to `tf.keras.Model`.

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
        stackwise_dilations=None,
        weights=None,
        input_shape=(None, None, 3),
        input_tensor=None,
        pooling=None,
        classes=None,
        classifier_activation="softmax",
        block_fn=Block,
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

        x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool1_pool")(x)

        num_stacks = len(stackwise_filters)
        if stackwise_dilations is None:
            stackwise_dilations = [1] * num_stacks

        stack_level_outputs = {}
        for stack_index in range(num_stacks):
            x = Stack(
                filters=stackwise_filters[stack_index],
                blocks=stackwise_blocks[stack_index],
                stride=stackwise_strides[stack_index],
                dilation=stackwise_dilations[stack_index],
                block_fn=block_fn,
                first_shortcut=block_fn == Block or stack_index > 0,
                stack_index=stack_index,
            )(x)
            stack_level_outputs[stack_index + 2] = x

        x = layers.BatchNormalization(axis=BN_AXIS, epsilon=BN_EPSILON, name="post_bn")(x)
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

        super().__init__(inputs, x, **kwargs)

        # All references to `self` below this line.
        if weights is not None:
            self.load_weights(weights)
        # Set this private attribute for recreate backbone model with outputs at each of the
        # resolution level.
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
        self.block_fn = Block

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stackwise_filters": self.stackwise_filters,
                "stackwise_blocks": self.stackwise_blocks,
                "stackwise_strides": self.stackwise_strides,
                "include_rescaling": self.include_rescaling,
                "include_top": self.include_top,
                "stackwise_dilations": self.stackwise_dilations,
                "input_tensor": self.input_tensor,
                "pooling": self.pooling,
                "classes": self.classes,
                "classifier_activation": self.classifier_activation,
                "block_fn": keras.utils.get_registered_name(self.block_fn),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        block_class_name = config["block_fn"]
        config["block_fn"] = keras.utils.get_registered_object(block_class_name)
        return super().from_config(config)


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
        block_fn=BasicBlock,
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
        block_fn=BasicBlock,
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
        **kwargs,
    )


setattr(ResNet18V2, "__doc__", BASE_DOCSTRING.format(name="ResNet18V2"))
setattr(ResNet34V2, "__doc__", BASE_DOCSTRING.format(name="ResNet34V2"))
setattr(ResNet50V2, "__doc__", BASE_DOCSTRING.format(name="ResNet50V2"))
setattr(ResNet101V2, "__doc__", BASE_DOCSTRING.format(name="ResNet101V2"))
setattr(ResNet152V2, "__doc__", BASE_DOCSTRING.format(name="ResNet152V2"))
