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

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers

from keras_cv.models import utils

MODEL_CONFIGS = {
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

BASE_DOCSTRING = """Instantiates the {name} architecture.
    Reference:
        - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
        - [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (ECCV 2016)
    This function returns a Keras {name} model.

    The difference in Resnet and ResNetV2 rests in the structure of their
    individual building blocks. In ResNetV2, the batch normalization and
    ReLU activation preceed the convolution layers, as opposed to ResNetV1 where
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
            specified if `include_top` is True, and if no `weights` argument is
            specified.
        weights: one of `None` (random initialization), or a pretrained weight file
            path.
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


def Block(filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """A residual block.
    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.
    Returns:
      Output tensor for the residual block.
    """
    if name is None:
        name = f"v1_block_{backend.get_uid('v1_block')}"

    def apply(x):
        if conv_shortcut:
            shortcut = layers.Conv2D(
                4 * filters, 1, strides=stride, name=name + "_0_conv"
            )(x)
            shortcut = layers.BatchNormalization(
                axis=BN_AXIS, epsilon=1.001e-5, name=name + "_0_bn"
            )(shortcut)
        else:
            shortcut = x

        x = layers.Conv2D(filters, 1, strides=stride, name=name + "_1_conv")(x)
        x = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=1.001e-5, name=name + "_1_bn"
        )(x)
        x = layers.Activation("relu", name=name + "_1_relu")(x)

        x = layers.Conv2D(filters, kernel_size, padding="SAME", name=name + "_2_conv")(
            x
        )
        x = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=1.001e-5, name=name + "_2_bn"
        )(x)
        x = layers.Activation("relu", name=name + "_2_relu")(x)

        x = layers.Conv2D(4 * filters, 1, name=name + "_3_conv")(x)
        x = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=1.001e-5, name=name + "_3_bn"
        )(x)

        x = layers.Add(name=name + "_add")([shortcut, x])
        x = layers.Activation("relu", name=name + "_out")(x)
        return x

    return apply


def Stack(filters, blocks, stride=2, name=None):
    """A set of stacked residual blocks.
    Args:
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.
    Returns:
      Output tensor for the stacked blocks.
    """
    if name is None:
        name = f"v1_stack_{backend.get_uid('v1_stack')}"

    def apply(x):
        x = Block(filters, stride=stride, name=name + "_block1")(x)
        for i in range(2, blocks + 1):
            x = Block(filters, conv_shortcut=False, name=name + "_block" + str(i))(x)
        return x

    return apply


def ResNet(
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
    classes=None,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ResNet architecture.

    Args:
        stackwise_filters: number of filters for each stack in the model.
        stackwise_blocks: number of blocks for each stack in the model.
        stackwise_strides: stride for each stack in the model.
        include_rescaling: whether or not to Rescale the inputs. If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
            name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
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
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.
        **kwargs: Pass-through keyword arguments to `tf.keras.Model`.

    Returns:
      A `keras.Model` instance.
    """
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
        64, 7, strides=2, use_bias=True, padding="same", name="conv1_conv"
    )(x)

    x = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name="conv1_bn")(x)
    x = layers.Activation("relu", name="conv1_relu")(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool1_pool")(x)

    num_stacks = len(stackwise_filters)

    for stack_index in range(num_stacks):
        x = Stack(
            filters=stackwise_filters[stack_index],
            blocks=stackwise_blocks[stack_index],
            stride=stackwise_strides[stack_index],
        )(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = layers.Dense(classes, activation=classifier_activation, name="predictions")(
            x
        )
    else:
        if pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D(name="max_pool")(x)

    # Create model.
    model = tf.keras.Model(inputs, x, name=name, **kwargs)

    if weights is not None:
        model.load_weights(weights)

    return model


def ResNet50(
    include_rescaling,
    include_top,
    classes=None,
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
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ResNet101(
    include_rescaling,
    include_top,
    classes=None,
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
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ResNet152(
    include_rescaling,
    include_top,
    classes=None,
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
        classes=classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


setattr(ResNet50, "__doc__", BASE_DOCSTRING.format(name="ResNet50"))
setattr(ResNet101, "__doc__", BASE_DOCSTRING.format(name="ResNet101"))
setattr(ResNet152, "__doc__", BASE_DOCSTRING.format(name="ResNet152"))
