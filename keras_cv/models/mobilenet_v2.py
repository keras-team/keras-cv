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

"""MobileNet V2 models for KerasCV.

References:
    - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) (CVPR 2018)
    - [Based on the original keras.applications MobileNetV2](https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet_v2.py)
"""

import tensorflow as tf
from keras import backend
from keras import layers

from keras_cv.models import utils

BASE_DOCSTRING = """Instantiates the {name} architecture.

    References:
        - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) (CVPR 2018)
        - [Based on the original keras.applications MobileNetv2](https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet_v2.py)

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
            depth multiplier in the MobileNetV2 paper, but the name is kept for
            consistency with MobileNetV1 in Keras.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                are used at each layer.
        classifier_activation: the activation function to use, defaults to softmax.
        name: (Optional) name to pass to the model. Defaults to "{name}".

    Returns:
        A `keras.Model` instance.
"""


BN_AXIS = 3


def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    Args:
      inputs: Input tensor.
      kernel_size: An integer or tuple/list of 2 integers.

    Returns:
      the tuple for zero padding.
    """
    img_dim = 1
    input_size = backend.int_shape(inputs)[img_dim : (img_dim + 2)]
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return (
        (correct[0] - adjust[0], correct[0]),
        (correct[1] - adjust[1], correct[1]),
    )


def InvertedResBlock(filters_in, filters_out, strides, expand_ratio, name=None):
    """An Inverted Residual Block.

    Args:
        filters_in: integer, the number of input filters.
        filters_out: integer, the number of output filters.
        strides: integer, the stride of the convolution.
        expand_ratio: integer, scaling coefficient for the input filters.
        name: string, layer label.

    Returns:
        a function that takes an input Tensor representing a InvertedResBlock.
    """
    if name is None:
        name = f"block{backend.get_uid('inverted_res_block')}_"

    def apply(inputs):
        shortcut = inputs

        if expand_ratio != 1:
            # Expansion phase
            x = layers.Conv2D(
                filters_in * expand_ratio,
                kernel_size=1,
                padding="same",
                use_bias=False,
                name=name + "expand_conv",
            )(inputs)
            x = layers.BatchNormalization(
                axis=BN_AXIS, epsilon=1e-3, momentum=0.999, name=name + "expand_bn"
            )(x)
            x = layers.ReLU(6.0, name=name + "expand_relu6")(x)
        else:
            x = inputs

        # Depthwise 3x3 convolution
        if strides == 2:
            x = layers.ZeroPadding2D(
                padding=correct_pad(x, 3), name=name + "dwconv_pad"
            )(x)
        x = layers.DepthwiseConv2D(
            kernel_size=3,
            strides=strides,
            padding="same" if strides == 1 else "valid",
            use_bias=False,
            name=name + "dwconv",
        )(x)
        x = layers.BatchNormalization(
            axis=BN_AXIS,
            epsilon=1e-3,
            momentum=0.999,
            name=name + "bn",
        )(x)
        x = layers.ReLU(6.0, name=name + "relu6")(x)

        # Project with a pointwise 1x1 convolution
        x = layers.Conv2D(
            filters_out,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name=name + "project_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=1e-3, momentum=0.999, name=name + "project_bn"
        )(x)
        if strides == 1 and filters_in == filters_out:
            x = layers.Add(name=name + "add")([shortcut, x])

        return x

    return apply


def MobileNetV2(
    *,
    include_rescaling,
    include_top,
    classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    alpha=1.0,
    classifier_activation="softmax",
    name="MobileNetV2",
    **kwargs,
):
    """Instantiates the MobileNetV2 architecture.

    References:
        - [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf) (CVPR 2018)
        - [Based on the original keras.applications MobileNetv2](https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet_v2.py)

    This function returns a Keras MobileNetV2 model.

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
        classifier_activation: the activation function to use, defaults to softmax.
        name: (Optional) name to pass to the model. Defaults to "MobileNetV2".

    Returns:
        A `keras.Model` instance.

    Raises:
        ValueError: if `weights` represents an invalid path to weights file and is not
            None.
        ValueError: if `include_top` is True and `classes` is not specified.
    """
    if weights and not tf.io.gfile.exists(weights):
        raise ValueError(
            "The `weights` argument should be either `None` or the path to the "
            f"weights file to be loaded. Weights file not found at location: {weights}"
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

    img_input = utils.parse_model_inputs(input_shape, input_tensor)

    def round_filters(filters, divisor=8):
        """Round number of filters based on alpha."""
        filters *= alpha
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    # Build stem
    x = img_input

    if include_rescaling:
        # Use common rescaling strategy across keras_cv
        x = layers.Rescaling(1.0 / 255.0)(x)

    x = layers.Conv2D(
        round_filters(32),
        kernel_size=3,
        strides=(2, 2),
        padding="same",
        use_bias=False,
        name="stem_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=BN_AXIS, epsilon=1e-3, momentum=0.999, name="stem_bn"
    )(x)
    x = layers.ReLU(6.0, name="stem_relu6")(x)

    # Build blocks
    x = InvertedResBlock(round_filters(32), round_filters(16), 1, 1)(x)

    x = InvertedResBlock(round_filters(16), round_filters(24), 2, 6)(x)
    x = InvertedResBlock(round_filters(24), round_filters(24), 1, 6)(x)

    x = InvertedResBlock(round_filters(24), round_filters(32), 2, 6)(x)
    x = InvertedResBlock(round_filters(32), round_filters(32), 1, 6)(x)
    x = InvertedResBlock(round_filters(32), round_filters(32), 1, 6)(x)

    x = InvertedResBlock(round_filters(32), round_filters(64), 2, 6)(x)
    x = InvertedResBlock(round_filters(64), round_filters(64), 1, 6)(x)
    x = InvertedResBlock(round_filters(64), round_filters(64), 1, 6)(x)
    x = InvertedResBlock(round_filters(64), round_filters(64), 1, 6)(x)

    x = InvertedResBlock(round_filters(64), round_filters(96), 1, 6)(x)
    x = InvertedResBlock(round_filters(96), round_filters(96), 1, 6)(x)
    x = InvertedResBlock(round_filters(96), round_filters(96), 1, 6)(x)

    x = InvertedResBlock(round_filters(96), round_filters(160), 2, 6)(x)
    x = InvertedResBlock(round_filters(160), round_filters(160), 1, 6)(x)
    x = InvertedResBlock(round_filters(160), round_filters(160), 1, 6)(x)

    x = InvertedResBlock(round_filters(160), round_filters(320), 1, 6)(x)

    # Build top
    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we increase the number of output
    # channels.
    last_block_filters = round_filters(1280) if alpha > 1.0 else 1280

    x = layers.Conv2D(
        last_block_filters, kernel_size=1, use_bias=False, name="top_conv"
    )(x)
    x = layers.BatchNormalization(
        axis=BN_AXIS, epsilon=1e-3, momentum=0.999, name="top_bn"
    )(x)
    x = layers.ReLU(6.0, name="top_relu6")(x)

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

    inputs = img_input

    # Create model
    model = tf.keras.Model(inputs, x, name=name, **kwargs)

    # Load weights
    if weights is not None:
        model.load_weights(weights)

    return model


MobileNetV2.__doc__ = BASE_DOCSTRING.format(name="MobileNetV2")
