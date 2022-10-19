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

"""MobileNet v3 models for KerasCV.

References:
    - [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf) (ICCV 2019)
    - [Based on the original keras.applications MobileNetv3](https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet_v3.py)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras.utils import custom_object_scope

from keras_cv import layers as cv_layers
from keras_cv.models import utils

channel_axis = -1

BASE_DOCSTRING = """Instantiates the {name} architecture.

    References:
        - [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
        - [Based on the Original keras.applications MobileNetv3](https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet_v3.py)

    This function returns a Keras {name} model.

    For transfer learning use cases, make sure to read the [guide to transfer
        learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        include_rescaling: whether or not to Rescale the inputs.If set to True,
            inputs will be passed through a `Rescaling(scale=1 / 255)`
            layer, defaults to True.
        include_top: whether to include the fully-connected layer at the top of the
            network.  If provided, `classes` must be provided.
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
        alpha: controls the width of the network. This is known as the
            depth multiplier in the MobileNetV3 paper, but the name is kept for
            consistency with MobileNetV1 in Keras.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                are used at each layer.
        minimalistic: in addition to large and small models this module also
            contains so-called minimalistic models, these models have the same
            per-layer dimensions characteristic as MobilenetV3 however, they don't
            utilize any of the advanced blocks (squeeze-and-excite units, hard-swish,
            and 5x5 convolutions). While these models are less efficient on CPU, they
            are much more performant on GPU/DSP.
        dropout_rate: a float between 0 and 1 denoting the fraction of input units to
            drop, defaults to 0.2.
        classifier_activation: the activation function to use, defaults to softmax.
        name: (Optional) name to pass to the model. Defaults to "{name}".

    Returns:
        A `keras.Model` instance.
"""


def depth(x, divisor=8, min_value=None):
    """Ensure that all layers have a channel number that is divisble by the `divisor`.

    Args:
        x: input value.
        divisor: integer, the value by which a channel number should be divisble,
            defaults to 8.
        min_value: float, minimum value for the new tensor.

    Returns:
        the updated value of the input.
    """

    if min_value is None:
        min_value = divisor

    new_x = max(min_value, int(x + divisor / 2) // divisor * divisor)

    # make sure that round down does not go down by more than 10%.
    if new_x < 0.9 * x:
        new_x += divisor
    return new_x


def HardSigmoid(name=None):
    """The Hard Sigmoid function.

    Args:
        name: string, layer label.

    Returns:
        a function that takes an input Tensor representing a HardSigmoid layer.
    """
    if name is None:
        name = f"hard_sigmoid_{backend.get_uid('hard_sigmoid')}"

    activation = layers.ReLU(6.0)

    def apply(x):
        return activation(x + 3.0) * (1.0 / 6.0)

    return apply


def HardSwish(name=None):
    """The Hard Swish function.

    Args:
        name: string, layer label.

    Returns:
        a function that takes an input Tensor representing a HardSwish layer.
    """
    if name is None:
        name = f"hard_swish_{backend.get_uid('hard_swish')}"

    hard_sigmoid = HardSigmoid()
    multiply_layer = layers.Multiply()

    def apply(x):
        return multiply_layer([x, hard_sigmoid(x)])

    return apply


def InvertedResBlock(
    expansion, filters, kernel_size, stride, se_ratio, activation, block_id, name=None
):
    """An Inverted Residual Block.

    Args:
        expansion: integer, the expansion ratio, multiplied with infilters to get the
            minimum value passed to depth.
        filters: integer, number of filters for convolution layer.
        kernel_size: integer, the kernel size for DpethWise Convolutions.
        stride: integer, the stride length for DpethWise Convolutions.
        se_ratio: float, ratio for bottleneck filters. Number of bottleneck
            filters = filters * se_ratio.
        activation: the activation layer to use.
        block_id: integer, a unique identification if you want to use expanded
            convolutions.
        name: string, layer label.

    Returns:
        a function that takes an input Tensor representing a InvertedResBlock.
    """
    if name is None:
        name = f"inverted_res_block_{backend.get_uid('inverted_res_block')}"

    def apply(x):
        shortcut = x
        prefix = "expanded_conv/"
        infilters = backend.int_shape(x)[channel_axis]

        if block_id:
            prefix = f"expanded_conv_{block_id}"

            x = layers.Conv2D(
                depth(infilters * expansion),
                kernel_size=1,
                padding="same",
                use_bias=False,
                name=prefix + "expand",
            )(x)
            x = layers.BatchNormalization(
                axis=channel_axis,
                epsilon=1e-3,
                momentum=0.999,
                name=prefix + "expand/BatchNorm",
            )(x)
            x = activation(x)

        x = layers.DepthwiseConv2D(
            kernel_size,
            strides=stride,
            padding="same" if stride == 1 else "valid",
            use_bias=False,
            name=prefix + "depthwise",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + "depthwise/BatchNorm",
        )(x)
        x = activation(x)

        if se_ratio:
            with custom_object_scope({"hard_sigmoid": HardSigmoid()}):
                x = cv_layers.SqueezeAndExcite2D(
                    filters=depth(infilters * expansion),
                    ratio=se_ratio,
                    squeeze_activation="relu",
                    excite_activation="hard_sigmoid",
                )(x)

        x = layers.Conv2D(
            filters,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name=prefix + "project",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + "project/BatchNorm",
        )(x)

        if stride == 1 and infilters == filters:
            x = layers.Add(name=prefix + "Add")([shortcut, x])

        return x

    return apply


def MobileNetV3(
    stack_fn,
    last_point_ch,
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    alpha=1.0,
    minimalistic=True,
    dropout_rate=0.2,
    classifier_activation="softmax",
    name="MobileNetV3",
    **kwargs,
):
    """Instantiates the MobileNetV3 architecture.

    References:
        - [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf) (ICCV 2019)
        - [Based on the Original keras.applications MobileNetv3](https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet_v3.py)

    This function returns a Keras MobileNetV3 model.

    For transfer learning use cases, make sure to read the [guide to transfer
        learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        stack_fn: a function that returns tensors passed through Inverted
            Residual Blocks.
        last_point_ch: the number of filters for the convolution layer.
        include_rescaling: whether or not to Rescale the inputs.If set to True,
            inputs will be passed through a `Rescaling(scale=1 / 255)`
            layer, defaults to True.
        include_top: whether to include the fully-connected layer at the top of the
            network.  If provided, `classes` must be provided.
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
        alpha: controls the width of the network. This is known as the
            depth multiplier in the MobileNetV3 paper, but the name is kept for
            consistency with MobileNetV1 in Keras.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                are used at each layer.
        minimalistic: in addition to large and small models this module also
            contains so-called minimalistic models, these models have the same
            per-layer dimensions characteristic as MobilenetV3 however, they don't
            utilize any of the advanced blocks (squeeze-and-excite units, hard-swish,
            and 5x5 convolutions). While these models are less efficient on CPU, they
            are much more performant on GPU/DSP.
        dropout_rate: a float between 0 and 1 denoting the fraction of input units to
            drop, defaults to 0.2.
        classifier_activation: the activation function to use, defaults to softmax.

        name: (Optional) name to pass to the model. Defaults to "MobileNetV3".

    Returns:
        A `keras.Model` instance.

    Raises:
        ValueError: if `weights` represents an invalid path to weights file and is not
            None.
        ValueError: if `include_top` is True and `classes` is not specified.
    """
    if weights and not tf.io.gfile.exists(weights):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` or the path to the weights file to be loaded. "
            f"Weights file not found at location: {weights}"
        )

    if include_top and not classes:
        raise ValueError(
            "If `include_top` is True, "
            "you should specify `classes`. "
            f"Received: classes={classes}"
        )

    if minimalistic:
        kernel = 3
        activation = layers.ReLU()
        se_ratio = None
    else:
        kernel = 5
        activation = HardSwish()
        se_ratio = 0.25

    inputs = utils.parse_model_inputs(input_shape, input_tensor)

    x = inputs

    if include_rescaling:
        x = layers.Rescaling(scale=1 / 255)(x)

    x = layers.Conv2D(
        16,
        kernel_size=3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        name="Conv",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name="Conv/BatchNorm"
    )(x)
    x = activation(x)

    x = stack_fn(x, kernel, activation, se_ratio)

    last_conv_ch = depth(backend.int_shape(x)[channel_axis] * 6)

    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_point_ch = depth(last_point_ch * alpha)
    x = layers.Conv2D(
        last_conv_ch,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name="Conv_1",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3, momentum=0.999, name="Conv_1/BatchNorm"
    )(x)
    x = activation(x)
    if include_top:
        x = layers.GlobalAveragePooling2D(keepdims=True)(x)
        x = layers.Conv2D(
            last_point_ch,
            kernel_size=1,
            padding="same",
            use_bias=True,
            name="Conv_2",
        )(x)
        x = activation(x)

        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv2D(classes, kernel_size=1, padding="same", name="Logits")(x)
        x = layers.Flatten()(x)
        x = layers.Activation(activation=classifier_activation, name="Predictions")(x)
    elif pooling == "avg":
        x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    elif pooling == "max":
        x = layers.GlobalMaxPooling2D(name="max_pool")(x)

    model = keras.Model(inputs, x, name=name, **kwargs)

    if weights is not None:
        model.load_weights(weights)
    return model


def MobileNetV3Small(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    alpha=1.0,
    minimalistic=False,
    dropout_rate=0.2,
    classifier_activation="softmax",
    name="MobileNetV3Small",
    **kwargs,
):
    def stack_fn(x, kernel, activation, se_ratio):

        x = InvertedResBlock(1, depth(16 * alpha), 3, 2, se_ratio, layers.ReLU(), 0)(x)
        x = InvertedResBlock(
            72.0 / 16, depth(24 * alpha), 3, 2, None, layers.ReLU(), 1
        )(x)
        x = InvertedResBlock(
            88.0 / 24, depth(24 * alpha), 3, 1, None, layers.ReLU(), 2
        )(x)
        x = InvertedResBlock(4, depth(40 * alpha), kernel, 2, se_ratio, activation, 3)(
            x
        )
        x = InvertedResBlock(6, depth(40 * alpha), kernel, 1, se_ratio, activation, 4)(
            x
        )
        x = InvertedResBlock(6, depth(40 * alpha), kernel, 1, se_ratio, activation, 5)(
            x
        )
        x = InvertedResBlock(3, depth(48 * alpha), kernel, 1, se_ratio, activation, 6)(
            x
        )
        x = InvertedResBlock(3, depth(48 * alpha), kernel, 1, se_ratio, activation, 7)(
            x
        )
        x = InvertedResBlock(6, depth(96 * alpha), kernel, 2, se_ratio, activation, 8)(
            x
        )
        x = InvertedResBlock(6, depth(96 * alpha), kernel, 1, se_ratio, activation, 9)(
            x
        )
        x = InvertedResBlock(6, depth(96 * alpha), kernel, 1, se_ratio, activation, 10)(
            x
        )
        return x

    return MobileNetV3(
        stack_fn=stack_fn,
        last_point_ch=1024,
        include_rescaling=include_rescaling,
        include_top=include_top,
        classes=classes,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        alpha=alpha,
        minimalistic=minimalistic,
        dropout_rate=dropout_rate,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )


def MobileNetV3Large(
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    alpha=1.0,
    minimalistic=False,
    dropout_rate=0.2,
    classifier_activation="softmax",
    name="MobileNetV3Large",
    **kwargs,
):
    def stack_fn(x, kernel, activation, se_ratio):

        x = InvertedResBlock(1, depth(16 * alpha), 3, 1, None, layers.ReLU(), 0)(x)
        x = InvertedResBlock(4, depth(24 * alpha), 3, 2, None, layers.ReLU(), 1)(x)
        x = InvertedResBlock(3, depth(24 * alpha), 3, 1, None, layers.ReLU(), 2)(x)
        x = InvertedResBlock(
            3, depth(40 * alpha), kernel, 2, se_ratio, layers.ReLU(), 3
        )(x)
        x = InvertedResBlock(
            3, depth(40 * alpha), kernel, 1, se_ratio, layers.ReLU(), 4
        )(x)
        x = InvertedResBlock(
            3, depth(40 * alpha), kernel, 1, se_ratio, layers.ReLU(), 5
        )(x)
        x = InvertedResBlock(6, depth(80 * alpha), 3, 2, None, activation, 6)(x)
        x = InvertedResBlock(2.5, depth(80 * alpha), 3, 1, None, activation, 7)(x)
        x = InvertedResBlock(2.3, depth(80 * alpha), 3, 1, None, activation, 8)(x)
        x = InvertedResBlock(2.3, depth(80 * alpha), 3, 1, None, activation, 9)(x)
        x = InvertedResBlock(6, depth(112 * alpha), 3, 1, se_ratio, activation, 10)(x)
        x = InvertedResBlock(6, depth(112 * alpha), 3, 1, se_ratio, activation, 11)(x)
        x = InvertedResBlock(
            6, depth(160 * alpha), kernel, 2, se_ratio, activation, 12
        )(x)
        x = InvertedResBlock(
            6, depth(160 * alpha), kernel, 1, se_ratio, activation, 13
        )(x)
        x = InvertedResBlock(
            6, depth(160 * alpha), kernel, 1, se_ratio, activation, 14
        )(x)
        return x

    return MobileNetV3(
        stack_fn=stack_fn,
        last_point_ch=1280,
        include_rescaling=include_rescaling,
        include_top=include_top,
        classes=classes,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        alpha=alpha,
        minimalistic=minimalistic,
        dropout_rate=dropout_rate,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )


setattr(MobileNetV3Large, "__doc__", BASE_DOCSTRING.format(name="MobileNetV3Large"))
setattr(MobileNetV3Small, "__doc__", BASE_DOCSTRING.format(name="MobileNetV3Small"))
