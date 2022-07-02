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
  - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (CVPR 2015)
  - [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (ECCV 2016)
  - [Based on the original keras.applications ResNet](https://github.com/keras-team/keras/blob/master/keras/applications/py)
"""

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers

BN_AXIS = 3

BASE_DOCSTRING = """Instantiates the {name} architecture.
    Reference:
        - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
        - [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (ECCV 2016)
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


def ResNet(
    stack_fn,
    preact,
    include_rescaling,
    name="ResNet",
    include_top=True,
    weights="None",
    input_shape=(None, None, 3),
    pooling=None,
    num_classes=1000,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.
    Args:
      stack_fn: a function that returns output tensor for the
        stacked residual blocks.
      preact: whether to use pre-activation or not
        (True for ResNetV2, False for ResNet).
      include_rescaling: whether or not to Rescale the inputs.If set to True,
        inputs will be passed through a `Rescaling(1/255.0)` layer.
      name: string, model name.
      include_top: whether to include the fully-connected
        layer at the top of the network.
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
      num_classes: optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.
      **kwargs: For backwards compatibility only.

        Returns:
      A `keras.Model` instance.
    """
    if kwargs:
        raise ValueError("Unknown argument(s): %s" % (kwargs,))
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

    img_input = layers.Input(shape=input_shape)

    if include_rescaling:
        img_input = layers.Rescaling(1 / 255.0)(img_input)

    x = layers.Conv2D(
        64, 7, strides=2, use_bias=True, padding="same", name="conv1_conv"
    )(img_input)

    if not preact:
        x = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name="conv1_bn")(
            x
        )
        x = layers.Activation("relu", name="conv1_relu")(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool1_pool")(x)

    x = stack_fn(x)

    if preact:
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

    inputs = img_input

    # Create model.
    model = tf.keras.Model(inputs, x, name=name)

    if weights is not None:
        model.load_weights(weights)

    return model


# v1 block
def V1Block(filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
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


def V1Stack(filters, blocks, stride1=2, name=None):
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
        x = V1Block(filters, stride=stride1, name=name + "_block1")(x)
        for i in range(2, blocks + 1):
            x = V1Block(filters, conv_shortcut=False, name=name + "_block" + str(i))(x)
        return x

    return apply


def V2Block(filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    """A residual block.
    Args:
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default False, use convolution shortcut if True,
          otherwise identity shortcut.
        name: string, block label.
    Returns:
      Output tensor for the residual block.
    """
    if name is None:
        name = f"v2_block_{backend.get_uid('v2_block')}"

    def apply(x):
        preact = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=1.001e-5, name=name + "_preact_bn"
        )(x)

        preact = layers.Activation("relu", name=name + "_preact_relu")(preact)

        if conv_shortcut:
            shortcut = layers.Conv2D(
                4 * filters, 1, strides=stride, name=name + "_0_conv"
            )(preact)
        else:
            shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

        x = layers.Conv2D(filters, 1, strides=1, use_bias=False, name=name + "_1_conv")(
            preact
        )
        x = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=1.001e-5, name=name + "_1_bn"
        )(x)
        x = layers.Activation("relu", name=name + "_1_relu")(x)

        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + "_2_pad")(x)
        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=stride,
            use_bias=False,
            name=name + "_2_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=1.001e-5, name=name + "_2_bn"
        )(x)
        x = layers.Activation("relu", name=name + "_2_relu")(x)

        x = layers.Conv2D(4 * filters, 1, name=name + "_3_conv")(x)
        x = layers.Add(name=name + "_out")([shortcut, x])
        return x

    return apply


# v2 stack
def V2Stack(filters, blocks, stride=2, name=None):
    """A set of stacked residual blocks.
    Args:
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride: default 2, stride of the first layer in the first block.
        name: string, stack label.
    Returns:
        Output tensor for the stacked blocks.
    """
    if name is None:
        name = f"v2_stack_{backend.get_uid('v2_stack')}"

    def apply(x):
        x = V2Block(filters, conv_shortcut=True, name=name + "_block1")(x)
        for i in range(2, blocks):
            x = V2Block(filters, name=name + "_block" + str(i))(x)
        x = V2Block(filters, stride=stride, name=name + "_block" + str(blocks))(x)
        return x

    return apply


def ResNet50(
    include_rescaling,
    include_top,
    num_classes=1000,
    weights="None",
    input_shape=(None, None, 3),
    pooling=None,
    classifier_activation="softmax",
    name="resnet50",
    **kwargs,
):
    """Instantiates the ResNet50 architecture."""

    def stack_fn(x):
        x = V1Stack(64, 3, stride1=1, name="conv2")(x)
        x = V1Stack(128, 4, name="conv3")(x)
        x = V1Stack(256, 6, name="conv4")(x)
        return V1Stack(512, 3, name="conv5")(x)

    return ResNet(
        stack_fn=stack_fn,
        preact=False,
        include_rescaling=include_rescaling,
        name=name,
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ResNet101(
    include_rescaling,
    include_top,
    num_classes=1000,
    weights="None",
    input_shape=(None, None, 3),
    pooling=None,
    classifier_activation="softmax",
    name="resnet101",
    **kwargs,
):
    """Instantiates the ResNet101 architecture."""

    def stack_fn(x):
        x = V1Stack(64, 3, stride1=1, name="conv2")(x)
        x = V1Stack(128, 4, name="conv3")(x)
        x = V1Stack(256, 23, name="conv4")(x)
        return V1Stack(512, 3, name="conv5")(x)

    return ResNet(
        stack_fn=stack_fn,
        preact=False,
        name=name,
        include_rescaling=include_rescaling,
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ResNet152(
    include_rescaling,
    include_top,
    num_classes=1000,
    weights="None",
    input_shape=(None, None, 3),
    pooling=None,
    classifier_activation="softmax",
    name="resnet152",
    **kwargs,
):
    """Instantiates the ResNet152 architecture."""

    def stack_fn(x):
        x = V1Stack(64, 3, stride1=1, name="conv2")(x)
        x = V1Stack(128, 8, name="conv3")(x)
        x = V1Stack(256, 36, name="conv4")(x)
        return V1Stack(512, 3, name="conv5")(x)

    return ResNet(
        stack_fn=stack_fn,
        preact=False,
        include_rescaling=include_rescaling,
        name=name,
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ResNet50V2(
    include_rescaling,
    include_top,
    num_classes=1000,
    weights="None",
    input_shape=(None, None, 3),
    pooling=None,
    classifier_activation="softmax",
    name="resnet50v2",
    **kwargs,
):
    """Instantiates the ResNet50V2 architecture."""

    def stack_fn(x):
        x = V2Stack(64, 3, name="conv2")(x)
        x = V2Stack(128, 4, name="conv3")(x)
        x = V2Stack(256, 6, name="conv4")(x)
        return V2Stack(512, 3, stride=1, name="conv5")(x)

    return ResNet(
        stack_fn=stack_fn,
        preact=True,
        include_rescaling=include_rescaling,
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )


def ResNet101V2(
    include_rescaling,
    include_top,
    num_classes=1000,
    weights="None",
    input_shape=(None, None, 3),
    pooling=None,
    classifier_activation="softmax",
    name="resnet101v2",
    **kwargs,
):
    """Instantiates the ResNet101V2 architecture."""

    def stack_fn(x):
        x = V2Stack(64, 3, name="conv2")(x)
        x = V2Stack(128, 4, name="conv3")(x)
        x = V2Stack(256, 23, name="conv4")(x)
        return V2Stack(512, 3, stride=1, name="conv5")(x)

    return ResNet(
        stack_fn=stack_fn,
        preact=True,
        include_rescaling=include_rescaling,
        name=name,
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ResNet152V2(
    include_rescaling,
    include_top,
    num_classes=1000,
    weights="None",
    input_shape=(None, None, 3),
    pooling=None,
    classifier_activation="softmax",
    name="resnet152v2",
    **kwargs,
):
    """Instantiates the ResNet152V2 architecture."""

    def stack_fn(x):
        x = V2Stack(64, 3, name="conv2")(x)
        x = V2Stack(128, 8, name="conv3")(x)
        x = V2Stack(256, 36, name="conv4")(x)
        return V2Stack(512, 3, stride=1, name="conv5")(x)

    return ResNet(
        stack_fn=stack_fn,
        preact=True,
        include_rescaling=include_rescaling,
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )


setattr(ResNet50, "__doc__", BASE_DOCSTRING.format(name="ResNet50"))
setattr(ResNet101, "__doc__", BASE_DOCSTRING.format(name="ResNet101"))
setattr(ResNet152, "__doc__", BASE_DOCSTRING.format(name="ResNet152"))

setattr(ResNet50V2, "__doc__", BASE_DOCSTRING.format(name="ResNet50V2"))
setattr(ResNet101V2, "__doc__", BASE_DOCSTRING.format(name="ResNet101V2"))
setattr(ResNet152V2, "__doc__", BASE_DOCSTRING.format(name="ResNet152V2"))
