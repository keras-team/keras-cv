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

"""DenseNet models for KerasCV.

Reference:
  - [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
  - [Based on the Original keras.applications DenseNet](https://github.com/keras-team/keras/blob/master/keras/applications/densenet.py)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers

BN_AXIS = 3

BASE_DOCSTRING = """Instantiates the {name} architecture.

    Reference:
        - [Densely Connected Convolutional Networks (CVPR 2017)](https://arxiv.org/abs/1608.06993)

    This function returns a Keras {name} model.

    For transfer learning use cases, make sure to read the [guide to transfer
        learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        include_rescaling: whether or not to Rescale the inputs.If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        include_top: whether to include the fully-connected layer at the top of the
            network.  If provided, num_classes must be provided.
        num_classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True, and if no `weights` argument is
            specified.
        weights: one of `None` (random initialization), or a pretrained weight file
            path.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be the 4D tensor output
                of the last convolutional block.
            - `avg` means that global average pooling will be applied to the output
                of the last convolutional block, and thus the output of the model will
                be a 2D tensor.
            - `max` means that global max pooling will be applied.
        name: (Optional) name to pass to the model.  Defaults to "{name}".

    Returns:
      A `keras.Model` instance.
"""


def DenseBlock(blocks, name=None):
    """A dense block.

    Args:
      blocks: integer, the number of building blocks.
      name: string, block label.

    Returns:
      a function that takes an input Tensor representing a DenseBlock.
    """
    if name is None:
        name = f"dense_block_{backend.get_uid('dense_block')}"

    def apply(x):
        for i in range(blocks):
            x = ConvBlock(32, name=f"{name}_block_{i}")(x)
        return x

    return apply


def TransitionBlock(reduction, name=None):
    """A transition block.

    Args:
      reduction: float, compression rate at transition layers.
      name: string, block label.

    Returns:
      a function that takes an input Tensor representing a TransitionBlock.
    """
    if name is None:
        name = f"transition_block_{backend.get_uid('transition_block')}"

    def apply(x):
        x = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=1.001e-5, name=f"{name}_bn"
        )(x)
        x = layers.Activation("relu", name=f"{name}_relu")(x)
        x = layers.Conv2D(
            int(backend.int_shape(x)[BN_AXIS] * reduction),
            1,
            use_bias=False,
            name=f"{name}_conv",
        )(x)
        x = layers.AveragePooling2D(2, strides=2, name=f"{name}_pool")(x)
        return x

    return apply


def ConvBlock(growth_rate, name=None):
    """A building block for a dense block.

    Args:
      growth_rate: float, growth rate at dense layers.
      name: string, block label.

    Returns:
      a function that takes an input Tensor representing a ConvBlock.
    """
    if name is None:
        name = f"conv_block_{backend.get_uid('conv_block')}"

    def apply(x):
        x1 = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=1.001e-5, name=f"{name}_0_bn"
        )(x)
        x1 = layers.Activation("relu", name=f"{name}_0_relu")(x1)
        x1 = layers.Conv2D(4 * growth_rate, 1, use_bias=False, name=f"{name}_1_conv")(
            x1
        )
        x1 = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=1.001e-5, name=f"{name}_1_bn"
        )(x1)
        x1 = layers.Activation("relu", name=f"{name}_1_relu")(x1)
        x1 = layers.Conv2D(
            growth_rate, 3, padding="same", use_bias=False, name=f"{name}_2_conv"
        )(x1)
        x = layers.Concatenate(axis=BN_AXIS, name=f"{name}_concat")([x, x1])
        return x

    return apply


def DenseNet(
    blocks,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    pooling=None,
    classifier_activation="softmax",
    name="DenseNet",
    **kwargs,
):
    """Instantiates the DenseNet architecture.

    Reference:
        - [Densely Connected Convolutional Networks (CVPR 2017)](https://arxiv.org/abs/1608.06993)

    This function returns a Keras DenseNet model.

    For transfer learning use cases, make sure to read the [guide to transfer
        learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        blocks: numbers of building blocks for the four dense layers.
        include_rescaling: whether or not to Rescale the inputs.If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        include_top: whether to include the fully-connected layer at the top of the
            network.  If provided, num_classes must be provided.
        num_classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True, and if no `weights` argument is
            specified.
        weights: one of `None` (random initialization), or a pretrained weight file
            path.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be the 4D tensor output
                of the last convolutional block.
            - `avg` means that global average pooling will be applied to the output
                of the last convolutional block, and thus the output of the model will
                be a 2D tensor.
            - `max` means that global max pooling will be applied.
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.
            When loading pretrained weights, `classifier_activation` can only
            be `None` or `"softmax"`.
        name: (Optional) name to pass to the model.  Defaults to "DenseNet".

    Returns:
      A `keras.Model` instance.
    """
    if weights and not tf.io.gfile.exists(weights):
        raise ValueError(
            "The `weights` argument should be either `None` or the path to the "
            "weights file to be loaded. Weights file not found at location: {weights}"
        )

    if include_top and not num_classes:
        raise ValueError(
            "If `include_top` is True, you should specify `num_classes`. "
            f"Received: num_classes={num_classes}"
        )

    inputs = layers.Input(shape=input_shape)

    x = inputs
    if include_rescaling:
        x = layers.Rescaling(1 / 255.0)(x)

    x = layers.Conv2D(
        64, 7, strides=2, use_bias=False, padding="same", name="conv1/conv"
    )(x)
    x = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name="conv1/bn")(x)
    x = layers.Activation("relu", name="conv1/relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool1")(x)

    x = DenseBlock(blocks[0], name="conv2")(x)
    x = TransitionBlock(0.5, name="pool2")(x)
    x = DenseBlock(blocks[1], name="conv3")(x)
    x = TransitionBlock(0.5, name="pool3")(x)
    x = DenseBlock(blocks[2], name="conv4")(x)
    x = TransitionBlock(0.5, name="pool4")(x)
    x = DenseBlock(blocks[3], name="conv5")(x)

    x = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name="bn")(x)
    x = layers.Activation("relu", name="relu")(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = layers.Dense(
            num_classes, activation=classifier_activation, name="predictions"
        )(x)
    elif pooling == "avg":
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    elif pooling == "max":
        x = layers.GlobalMaxPooling2D(name="max_pool")(x)

    model = keras.Model(inputs, x, name=name, **kwargs)

    if weights is not None:
        model.load_weights(weights)
    return model


def DenseNet121(
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    pooling=None,
    name="DenseNet121",
    **kwargs,
):
    return DenseNet(
        [6, 12, 24, 16],
        include_rescaling=include_rescaling,
        include_top=include_top,
        num_classes=num_classes,
        weights=weights,
        input_shape=input_shape,
        pooling=pooling,
        name=name,
        **kwargs,
    )


def DenseNet169(
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    pooling=None,
    name="DenseNet169",
    **kwargs,
):
    return DenseNet(
        [6, 12, 32, 32],
        include_rescaling=include_rescaling,
        include_top=include_top,
        num_classes=num_classes,
        weights=weights,
        input_shape=input_shape,
        pooling=pooling,
        name=name,
        **kwargs,
    )


def DenseNet201(
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    pooling=None,
    name="DenseNet201",
    **kwargs,
):
    return DenseNet(
        blocks=[6, 12, 48, 32],
        include_rescaling=include_rescaling,
        include_top=include_top,
        num_classes=num_classes,
        weights=weights,
        input_shape=input_shape,
        pooling=pooling,
        name=name,
        **kwargs,
    )


setattr(DenseNet121, "__doc__", BASE_DOCSTRING.format(name="DenseNet121"))
setattr(DenseNet169, "__doc__", BASE_DOCSTRING.format(name="DenseNet169"))
setattr(DenseNet201, "__doc__", BASE_DOCSTRING.format(name="DenseNet201"))
