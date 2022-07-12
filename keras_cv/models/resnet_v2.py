# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet v2 models for Keras.

Reference:
  - [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (CVPR 2016)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers

BN_AXIS = 3

BASE_DOCSTRING = """Instantiates the {name} architecture

  Reference:
  - [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (CVPR 2016)

  For transfer learning use cases, make sure to read the
  [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

  Args:
    include_rescaling: whether or not to Rescale the inputs.If set to True,
      inputs will be passed through a `Rescaling(1/255.0)` layer.
    include_top: whether to include the fully-connected
      layer at the top of the network.
    num_classes: optional number of classes to classify images into, only to be
       specified if `include_top` is True, and if no `weights` argument is
       specified.
    weights: one of `None` (random initialization),
      or the path to the weights file to be loaded.
    input_shape: optional shape tuple, defaults to (None, None, 3).
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
    classifier_activation: A `str` or callable. The activation function to use
      on the "top" layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.
      When loading pretrained weights, `classifier_activation` can only
      be `None` or `"softmax"`.
    name: (Optional) name to pass to the model. Defaults to "ResNetV2"


  Returns:
    A `keras.Model` instance.
"""


def stack(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.
    Args:
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.
    Returns:
        Output tensor for the stacked blocks.
    """
    x = block(x, filters, conv_shortcut=True, name=name + "_block1")
    for i in range(2, blocks):
        x = block(x, filters, name=name + "_block" + str(i))
    x = block(x, filters, stride=stride1, name=name + "_block" + str(blocks))
    return x


def block(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    """A residual block.
    Args:
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default False, use convolution shortcut if True,
          otherwise identity shortcut.
        name: string, block label.
    Returns:
      Output tensor for the residual block.
    """
    BN_AXIS = 3 if backend.image_data_format() == "channels_last" else 1

    preact = layers.BatchNormalization(
        axis=BN_AXIS, epsilon=1.001e-5, name=name + "_preact_bn"
    )(x)
    preact = layers.Activation("relu", name=name + "_preact_relu")(preact)

    if conv_shortcut:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride, name=name + "_0_conv")(
            preact
        )
    else:
        shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = layers.Conv2D(filters, 1, strides=1, use_bias=False, name=name + "_1_conv")(
        preact
    )
    x = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name=name + "_1_bn")(
        x
    )
    x = layers.Activation("relu", name=name + "_1_relu")(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + "_2_pad")(x)
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        use_bias=False,
        name=name + "_2_conv",
    )(x)
    x = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name=name + "_2_bn")(
        x
    )
    x = layers.Activation("relu", name=name + "_2_relu")(x)

    x = layers.Conv2D(4 * filters, 1, name=name + "_3_conv")(x)
    x = layers.Add(name=name + "_out")([shortcut, x])
    return x


def ResNetV2(
    stack_fn,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    pooling=None,
    classifier_activation="softmax",
    name="ResNetV2",
    **kwargs,
):
    """Instantiates the ResNetV2 architecture.
    Args:
      stack_fn: a function that returns output tensor for the
        stacked residual blocks.
      include_rescaling: whether or not to Rescale the inputs.If set to True,
        inputs will be passed through a `Rescaling(1/255.0)` layer.
      include_top: whether to include the fully-connected
        layer at the top of the network.
      num_classes: optional number of classes to classify images into, only to be
         specified if `include_top` is True, and if no `weights` argument is
         specified.
      weights: one of `None` (random initialization),
        or the path to the weights file to be loaded.
      input_shape: optional shape tuple, defaults to (None, None, 3).
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
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.
      name: (Optional) name to pass to the model. Defaults to "ResNetV2"
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

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(x)
    x = layers.Conv2D(64, 7, strides=2, use_bias=True, name="conv1_conv")(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool1_pool")(x)

    x = stack_fn(x)

    x = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name="post_bn")(x)
    x = layers.Activation("relu", name="post_relu")(x)

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

    model = keras.Model(inputs, x, name=name, **kwargs)

    if weights is not None:
        model.load_weights(weights)

    return model


def ResNet50V2(
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    pooling=None,
    classifier_activation="softmax",
    name="ResNet50v2",
    **kwargs,
):
    def stack_fn(x):
        x = stack(x, 64, 3, name="conv2")
        x = stack(x, 128, 4, name="conv3")
        x = stack(x, 256, 6, name="conv4")
        return stack(x, 512, 3, stride1=1, name="conv5")

    return ResNetV2(
        stack_fn,
        include_rescaling,
        include_top,
        num_classes,
        weights,
        input_shape,
        pooling,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )


def ResNet101V2(
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    pooling=None,
    classifier_activation="softmax",
    name="ResNet101v2",
    **kwargs,
):
    def stack_fn(x):
        x = stack(x, 64, 3, name="conv2")
        x = stack(x, 128, 4, name="conv3")
        x = stack(x, 256, 23, name="conv4")
        return stack(x, 512, 3, stride1=1, name="conv5")

    return ResNetV2(
        stack_fn,
        include_rescaling,
        include_top,
        num_classes,
        weights,
        input_shape,
        pooling,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )


def ResNet152V2(
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    pooling=None,
    classifier_activation="softmax",
    name="ResNet152v2",
    **kwargs,
):
    def stack_fn(x):
        x = stack(x, 64, 3, name="conv2")
        x = stack(x, 128, 8, name="conv3")
        x = stack(x, 256, 36, name="conv4")
        return stack(x, 512, 3, stride1=1, name="conv5")

    return ResNetV2(
        stack_fn,
        include_rescaling,
        include_top,
        num_classes,
        weights,
        input_shape,
        pooling,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )


setattr(ResNet50V2, "__doc__", BASE_DOCSTRING.format(name="ResNet50V2"))
setattr(ResNet101V2, "__doc__", BASE_DOCSTRING.format(name="ResNet101V2"))
setattr(ResNet152V2, "__doc__", BASE_DOCSTRING.format(name="ResNet152V2"))
