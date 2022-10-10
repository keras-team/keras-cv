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
"""WideResNet models for KerasCV.
Reference:
  - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (CVPR 2015)
  - [Wide Residual Networks](https://arxiv.org/abs/1605.07146) (2016)
"""

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers

from keras_cv.models import utils

"""
Stackwise (groupwise) blocks depend on the desired depth: blocks = (depth-4) / 6
"""
MODEL_CONFIGS = {
    "WRN16_8": {
        "stackwise_filters": [16, 32, 64],
        "stackwise_blocks": [2, 2, 2],
        "k": 8,
        "l": 2,
        "stackwise_strides": [1, 2, 2],
    },
    "WRN16_10": {
        "stackwise_filters": [16, 32, 64],
        "stackwise_blocks": [2, 2, 2],
        "k": 10,
        "l": 2,
        "stackwise_strides": [1, 2, 2],
    },
    "WRN22_8": {
        "stackwise_filters": [16, 32, 64],
        "stackwise_blocks": [3, 3, 3],
        "k": 8,
        "l": 2,
        "stackwise_strides": [1, 2, 2],
    },
    "WRN22_10": {
        "stackwise_filters": [16, 32, 64],
        "stackwise_blocks": [3, 3, 3],
        "k": 10,
        "l": 2,
        "stackwise_strides": [1, 2, 2],
    },
    "WRN28_10": {
        "stackwise_filters": [16, 32, 64],
        "stackwise_blocks": [4, 4, 4],
        "k": 10,
        "l": 2,
        "stackwise_strides": [1, 2, 2],
    },
    "WRN28_12": {
        "stackwise_filters": [16, 32, 64],
        "stackwise_blocks": [4, 4, 4],
        "k": 12,
        "l": 2,
        "stackwise_strides": [1, 2, 2],
    },
    "WRN40_8": {
        "stackwise_filters": [16, 32, 64],
        "stackwise_blocks": [6, 6, 6],
        "k": 8,
        "l": 2,
        "stackwise_strides": [1, 2, 2],
    },
    "WRN50_2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [3, 4, 6, 3],
        "k": 2,
        "l": 2,
        "stackwise_strides": [1, 2, 2, 2],
    },
    "WRN101_2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [3, 4, 23, 3],
        "k": 2,
        "l": 2,
        "stackwise_strides": [1, 2, 2, 2],
    },
}

BN_AXIS = 3

BASE_DOCSTRING = """Instantiates the {name} architecture.
    Reference:
      - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (CVPR 2015)
      - [Wide Residual Networks](https://arxiv.org/abs/1605.07146) (2016)
    This function returns a Keras {name} model.
    WideResNets explore greater width with lower depths than original ResNets.
    The width is controlled by a width_factor (k) and the depth is controlled by a depth_factor (l).
    The original order of operations (conv-BN-ReLU) was changed to (BN-ReLU-conv) due to faster training.
    Dropout was introduced between successive Conv2D layers.
    For transfer learning use cases, make sure to read the [guide to transfer
        learning & fine-tuning](https://keras.io/guides/transfer_learning/).
    Args:
        include_rescaling: whether or not to Rescale the inputs.If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        include_top: whether to include the fully-connected layer at the top of the
            network.  If provided, classes must be provided.
        classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True.
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
        dropout: default 0.3, the dropout rate between Conv2D layers in each block.
        depth_factor: default 2, depth multiplier (number of Conv2D layers in each block).
        width_factor: default 1, width multiplier (k*filters in each Conv2D layer).
    Returns:
      A `keras.Model` instance.
"""


def WideDropoutBlock(
    filters,
    kernel_size=3,
    stride=1,
    conv_shortcut=True,
    depth_factor=2,
    width_factor=1,
    dropout=0.3,
    name=None,
):
    """A wide residual block with dropout.
    Args:
      x: input tensor.
      filters: integer, filters of the wide layer.
      kernel_size: default 3, kernel size of the wide layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      depth_factor: default 2, depth multiplier (number of Conv2D layers in each block)
      width_factor: default 1, width multiplier (k*filters in each Conv2D layer)
      name: string, block label.
    Returns:
      Output tensor for the residual block.
    """
    if name is None:
        name = f"block_{backend.get_uid('block')}"

    def apply(x):
        if conv_shortcut:
            shortcut = layers.BatchNormalization(
                axis=BN_AXIS, epsilon=1.001e-5, name=name + "_shortcut_bn"
            )(x)
            shortcut = layers.Activation("relu", name=name + "_shortcut_relu")(shortcut)
            shortcut = layers.Conv2D(
                k * filters,
                kernel_size,
                strides=stride,
                padding="same",
                use_bias=False,
                name=name + "_shortcut_conv",
                kernel_regularizer=tf.keras.regularizers.l2(0.0005),
            )(shortcut)
        else:
            shortcut = x

        for n in range(l):
            x = layers.BatchNormalization(
                axis=BN_AXIS, epsilon=1.001e-5, name=name + f"_{n}_bn"
            )(x)
            x = layers.Activation("relu", name=name + f"_{n}_relu")(x)
            if n == 0:
                x = layers.Conv2D(
                    k * filters,
                    kernel_size,
                    strides=stride,
                    padding="same",
                    use_bias=False,
                    name=name + "_0_strided_conv",
                    kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                )(x)
            else:
                x = layers.Conv2D(
                    k * filters,
                    kernel_size,
                    strides=1,
                    padding="same",
                    use_bias=False,
                    name=name + "_1_conv",
                    kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                )(x)

            if n % 2 == 0:
                x = layers.Dropout(dropout, name=name + "_0_dropout")(x)

        x = layers.Add(name=name + "_add")([shortcut, x])
        x = layers.Activation("relu", name=name + "_out")(x)
        return x

    return apply


def BottleneckBlock(
    filters,
    kernel_size=3,
    stride=1,
    conv_shortcut=True,
    depth_factor=2,
    width_factor=1,
    dropout=None,
    name=None,
):
    """A bottleneck residual block with a variable width factor.
    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      depth_factor: default 2, depth multiplier (number of Conv2D layers in each block)
      width_factor: default 1, width multiplier (k*filters in each Conv2D layer)
      name: string, block label.
    Returns:
      Output tensor for the residual block.
    """
    if name is None:
        name = f"block_{backend.get_uid('block')}"

    def apply(x):
        if conv_shortcut:
            shortcut = layers.BatchNormalization(
                axis=BN_AXIS, epsilon=1.001e-5, name=name + "_shortcut_bn"
            )(x)
            shortcut = layers.Activation("relu", name=name + "_shortcut_relu")(shortcut)
            shortcut = layers.Conv2D(
                4 * filters,
                1,
                strides=stride,
                padding="same",
                use_bias=False,
                name=name + "_shortcut_conv",
                kernel_regularizer=tf.keras.regularizers.l2(0.0005),
            )(shortcut)
        else:
            shortcut = x

        x = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=1.001e-5, name=name + "_0_bn"
        )(x)
        x = layers.Activation("relu", name=name + "_0_relu")(x)
        x = layers.Conv2D(
            filters,
            1,
            strides=stride,
            padding="same",
            use_bias=False,
            name=name + "_0_strided_conv",
            kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        )(x)

        x = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=1.001e-5, name=name + "_1_bn"
        )(x)
        x = layers.Activation("relu", name=name + "_1_relu")(x)
        x = layers.Conv2D(
            k * filters,
            kernel_size,
            strides=1,
            padding="same",
            use_bias=False,
            name=name + "_1_strided_conv",
            kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        )(x)

        x = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=1.001e-5, name=name + "_2_bn"
        )(x)
        x = layers.Activation("relu", name=name + "_2_relu")(x)
        x = layers.Conv2D(
            4 * filters,
            1,
            strides=1,
            padding="same",
            use_bias=False,
            name=name + "_2_conv",
            kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        )(x)

        x = layers.Add(name=name + "_add")([shortcut, x])
        x = layers.Activation("relu", name=name + "_out")(x)
        return x

    return apply


def Stack(
    filters,
    blocks,
    stride=2,
    name=None,
    block_fn=WideDropoutBlock,
    first_shortcut=True,
    depth_factor=2,
    width_factor=1,
    dropout=0.3,
):
    """A set of stacked wide residual blocks. Called 'groups' in the original paper.
    Args:
      filters: integer, filters of the layers in a block.
      blocks: integer, blocks in the stacked blocks.
      stride: default 2, stride of the first layer in the first block.
      name: string, stack label.
      block_fn: callable, `WideDropoutBlock` or `BottleneckBlock`, the block function to stack.
      first_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      depth_factor: default 2, depth multiplier (number of Conv2D layers in each block)
      width_factor: default 1, width multiplier (k*filters in each Conv2D layer)
      dropout: default 0.3, dropout rate in spatial dropout layers
    Returns:
      Output tensor for the stacked blocks.
    """
    if name is None:
        name = f"stack_{backend.get_uid('stack')}"

    def apply(x):
        x = block_fn(
            filters,
            stride=stride,
            conv_shortcut=first_shortcut,
            depth_factor=1,
            width_factor=k,
            dropout=dropout,
            name=name + "_conv1",
        )(x)
        for i in range(2, blocks + 1):
            x = block_fn(
                filters,
                conv_shortcut=False,
                depth_factor=l,
                width_factor=k,
                dropout=dropout,
                name=f"{name}_conv{i}",
            )(x)
        return x

    return apply


def WideResNet(
    stackwise_filters,
    stackwise_blocks,
    stackwise_strides,
    include_rescaling,
    include_top,
    width_factor=None,
    depth_factor=None,
    dropout=0.3,
    name="WideResNet",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
    classifier_activation="softmax",
    block_fn=WideDropoutBlock,
    **kwargs,
):
    """Instantiates the WideResNet architecture.
    Args:
        stackwise_filters: number of filters for each stack in the model.
        stackwise_blocks: number of blocks for each stack in the model.
        stackwise_strides: stride for each stack in the model.
        depth_factor: depth multiplier (number of Conv2D layers in each block).
        width_factor: width multiplier (k*filters in each Conv2D layer).
        dropout: the dropout rate between Conv2D layers in each block.
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
            into, only to be specified if `include_top` is True.
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.
        block_fn: callable, `WideDropoutBlock` or `BottleneckBlock`, the block function to stack.
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

    if block_fn == WideDropoutBlocwidth_factor:
        x = layers.Conv2D(
            16,
            3,
            strides=1,
            use_bias=False,
            padding="same",
            name="conv0_conv",
            kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        )(x)
    else:
        x = layers.Conv2D(
            64,
            7,
            strides=2,
            use_bias=False,
            padding="same",
            name="conv0_conv",
            kernel_regularizer=tf.keras.regularizers.l2(0.0005),
        )(x)
        x = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name="conv0_bn")(
            x
        )
        x = layers.Activation("relu", name="conv0_relu")(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool0_pool")(x)

    num_stacks = len(stackwise_filters)
    for stack_index in range(num_stacks):
        x = Stack(
            filters=stackwise_filters[stack_index],
            blocks=stackwise_blocks[stack_index],
            stride=stackwise_strides[stack_index],
            block_fn=block_fn,
            width_factor=k,
            depth_factor=l,
            dropout=dropout,
            first_shortcut=block_fn == WideDropoutBlock
            or BottleneckBlock
            or stack_index > 0,
            name=f"group{stack_index}",
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


def WideResNet16_8(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    width_factor=None,
    depth_factor=None,
    dropout=0.3,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="wideresnet16_8",
    **kwargs,
):
    """Instantiates the WideResnet16_8 architecture."""

    return WideResNet(
        stackwise_filters=MODEL_CONFIGS["WRN16_8"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["WRN16_8"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["WRN16_8"]["stackwise_strides"],
        width_factor=MODEL_CONFIGS["WRN16_8"]["k"],
        depth_factor=MODEL_CONFIGS["WRN16_8"]["l"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        name=name,
        weights=weights,
        dropout=dropout,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        block_fn=WideDropoutBlock,
        **kwargs,
    )


def WideResNet16_10(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    width_factor=None,
    depth_factor=None,
    dropout=0.3,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="wideresnet16_10",
    **kwargs,
):
    """Instantiates the WideResNet16_10 architecture."""

    return WideResNet(
        stackwise_filters=MODEL_CONFIGS["WRN16_10"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["WRN16_10"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["WRN16_10"]["stackwise_strides"],
        width_factor=MODEL_CONFIGS["WRN16_10"]["k"],
        depth_factor=MODEL_CONFIGS["WRN16_10"]["l"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        name=name,
        weights=weights,
        dropout=dropout,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        block_fn=WideDropoutBlock,
        **kwargs,
    )


def WideResNet22_8(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    width_factor=None,
    depth_factor=None,
    dropout=0.3,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="wideresnet22_8",
    **kwargs,
):
    """Instantiates the WideResNet22_8 architecture."""

    return WideResNet(
        stackwise_filters=MODEL_CONFIGS["WRN22_8"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["WRN22_8"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["WRN22_8"]["stackwise_strides"],
        width_factor=MODEL_CONFIGS["WRN22_8"]["k"],
        depth_factor=MODEL_CONFIGS["WRN22_8"]["l"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        name=name,
        weights=weights,
        dropout=dropout,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        block_fn=WideDropoutBlock,
        **kwargs,
    )


def WideResNet22_10(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    width_factor=None,
    depth_factor=None,
    dropout=0.3,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="wideresnet22_10",
    **kwargs,
):
    """Instantiates the WideResNet22_10 architecture."""

    return WideResNet(
        stackwise_filters=MODEL_CONFIGS["WRN22_10"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["WRN22_10"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["WRN22_10"]["stackwise_strides"],
        width_factor=MODEL_CONFIGS["WRN22_10"]["k"],
        depth_factor=MODEL_CONFIGS["WRN22_10"]["l"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        name=name,
        weights=weights,
        dropout=dropout,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        block_fn=WideDropoutBlock,
        **kwargs,
    )


def WideResNet28_10(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    width_factor=None,
    depth_factor=None,
    dropout=0.3,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="wideresnet28_10",
    **kwargs,
):
    """Instantiates the WideResNet28_10 architecture."""

    return WideResNet(
        stackwise_filters=MODEL_CONFIGS["WRN28_10"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["WRN28_10"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["WRN28_10"]["stackwise_strides"],
        width_factor=MODEL_CONFIGS["WRN28_10"]["k"],
        depth_factor=MODEL_CONFIGS["WRN28_10"]["l"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        name=name,
        weights=weights,
        dropout=dropout,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        block_fn=WideDropoutBlock,
        **kwargs,
    )


def WideResNet28_12(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    width_factor=None,
    depth_factor=None,
    dropout=0.3,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="wideresnet28_12",
    **kwargs,
):
    """Instantiates the WideResNet28_12 architecture."""

    return WideResNet(
        stackwise_filters=MODEL_CONFIGS["WRN28_12"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["WRN28_12"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["WRN28_12"]["stackwise_strides"],
        width_factor=MODEL_CONFIGS["WRN28_12"]["k"],
        depth_factor=MODEL_CONFIGS["WRN28_12"]["l"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        name=name,
        weights=weights,
        dropout=dropout,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        block_fn=WideDropoutBlock,
        **kwargs,
    )


def WideResNet40_8(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    width_factor=None,
    depth_factor=None,
    dropout=0.3,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="wideresnet40_8",
    **kwargs,
):
    """Instantiates the WideResNet40_8 architecture."""

    return WideResNet(
        stackwise_filters=MODEL_CONFIGS["WRN40_8"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["WRN40_8"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["WRN40_8"]["stackwise_strides"],
        width_factor=MODEL_CONFIGS["WRN40_8"]["k"],
        depth_factor=MODEL_CONFIGS["WRN40_8"]["l"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        name=name,
        weights=weights,
        dropout=dropout,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        block_fn=WideDropoutBlock,
        **kwargs,
    )


def WideResNet50_2(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    width_factor=None,
    depth_factor=None,
    dropout=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="wideresnet50_2",
    **kwargs,
):
    """Instantiates the WideResNet50_2 architecture."""

    return WideResNet(
        stackwise_filters=MODEL_CONFIGS["WRN50_2"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["WRN50_2"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["WRN50_2"]["stackwise_strides"],
        width_factor=MODEL_CONFIGS["WRN50_2"]["k"],
        depth_factor=MODEL_CONFIGS["WRN50_2"]["l"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        name=name,
        weights=weights,
        dropout=dropout,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        block_fn=BottleneckBlock,
        **kwargs,
    )


def WideResNet101_2(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    width_factor=None,
    depth_factor=None,
    dropout=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="wideresnet101_2",
    **kwargs,
):
    """Instantiates the WideResNet101_2 architecture."""

    return WideResNet(
        stackwise_filters=MODEL_CONFIGS["WRN101_2"]["stackwise_filters"],
        stackwise_blocks=MODEL_CONFIGS["WRN101_2"]["stackwise_blocks"],
        stackwise_strides=MODEL_CONFIGS["WRN101_2"]["stackwise_strides"],
        width_factor=MODEL_CONFIGS["WRN101_2"]["k"],
        depth_factor=MODEL_CONFIGS["WRN101_2"]["l"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        name=name,
        weights=weights,
        dropout=dropout,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
        block_fn=BottleneckBlock,
        **kwargs,
    )


setattr(WideResNet16_8, "__doc__", BASE_DOCSTRING.format(name="WideResNet16_8"))
setattr(WideResNet16_10, "__doc__", BASE_DOCSTRING.format(name="WideResNet16_10"))
setattr(WideResNet22_8, "__doc__", BASE_DOCSTRING.format(name="WideResNet22_8"))
setattr(WideResNet22_10, "__doc__", BASE_DOCSTRING.format(name="WideResNet22_10"))
setattr(WideResNet28_10, "__doc__", BASE_DOCSTRING.format(name="WideResNet28_10"))
setattr(WideResNet28_12, "__doc__", BASE_DOCSTRING.format(name="WideResNet28_12"))
setattr(WideResNet40_8, "__doc__", BASE_DOCSTRING.format(name="WideResNet40_8"))
setattr(WideResNet50_2, "__doc__", BASE_DOCSTRING.format(name="WideResNet50_2"))
setattr(WideResNet101_2, "__doc__", BASE_DOCSTRING.format(name="WideResNet101_2"))
