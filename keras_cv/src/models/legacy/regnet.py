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
"""RegNet models for KerasCV.
References:
    - [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678) (CVPR 2020)
    - [Based on the Original keras.applications RegNet](https://github.com/keras-team/keras/blob/master/keras/applications/regnet.py)
"""  # noqa: E501

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers

from keras_cv.src.layers import SqueezeAndExcite2D
from keras_cv.src.models.legacy import utils
from keras_cv.src.models.legacy.weights import parse_weights

# The widths and depths are deduced from a quantized linear function. For
# more information, please refer to "Designing Network Design Spaces" by
# Radosavovic et al.

# BatchNorm momentum and epsilon values taken from original implementation.

MODEL_CONFIGS = {
    "x002": {
        "depths": [1, 1, 4, 7],
        "widths": [24, 56, 152, 368],
        "group_width": 8,
        "default_size": 224,
        "block_type": "X",
    },
    "x004": {
        "depths": [1, 2, 7, 12],
        "widths": [32, 64, 160, 384],
        "group_width": 16,
        "default_size": 224,
        "block_type": "X",
    },
    "x006": {
        "depths": [1, 3, 5, 7],
        "widths": [48, 96, 240, 528],
        "group_width": 24,
        "default_size": 224,
        "block_type": "X",
    },
    "x008": {
        "depths": [1, 3, 7, 5],
        "widths": [64, 128, 288, 672],
        "group_width": 16,
        "default_size": 224,
        "block_type": "X",
    },
    "x016": {
        "depths": [2, 4, 10, 2],
        "widths": [72, 168, 408, 912],
        "group_width": 24,
        "default_size": 224,
        "block_type": "X",
    },
    "x032": {
        "depths": [2, 6, 15, 2],
        "widths": [96, 192, 432, 1008],
        "group_width": 48,
        "default_size": 224,
        "block_type": "X",
    },
    "x040": {
        "depths": [2, 5, 14, 2],
        "widths": [80, 240, 560, 1360],
        "group_width": 40,
        "default_size": 224,
        "block_type": "X",
    },
    "x064": {
        "depths": [2, 4, 10, 1],
        "widths": [168, 392, 784, 1624],
        "group_width": 56,
        "default_size": 224,
        "block_type": "X",
    },
    "x080": {
        "depths": [2, 5, 15, 1],
        "widths": [80, 240, 720, 1920],
        "group_width": 120,
        "default_size": 224,
        "block_type": "X",
    },
    "x120": {
        "depths": [2, 5, 11, 1],
        "widths": [224, 448, 896, 2240],
        "group_width": 112,
        "default_size": 224,
        "block_type": "X",
    },
    "x160": {
        "depths": [2, 6, 13, 1],
        "widths": [256, 512, 896, 2048],
        "group_width": 128,
        "default_size": 224,
        "block_type": "X",
    },
    "x320": {
        "depths": [2, 7, 13, 1],
        "widths": [336, 672, 1344, 2520],
        "group_width": 168,
        "default_size": 224,
        "block_type": "X",
    },
    "y002": {
        "depths": [1, 1, 4, 7],
        "widths": [24, 56, 152, 368],
        "group_width": 8,
        "default_size": 224,
        "block_type": "Y",
    },
    "y004": {
        "depths": [1, 3, 6, 6],
        "widths": [48, 104, 208, 440],
        "group_width": 8,
        "default_size": 224,
        "block_type": "Y",
    },
    "y006": {
        "depths": [1, 3, 7, 4],
        "widths": [48, 112, 256, 608],
        "group_width": 16,
        "default_size": 224,
        "block_type": "Y",
    },
    "y008": {
        "depths": [1, 3, 8, 2],
        "widths": [64, 128, 320, 768],
        "group_width": 16,
        "default_size": 224,
        "block_type": "Y",
    },
    "y016": {
        "depths": [2, 6, 17, 2],
        "widths": [48, 120, 336, 888],
        "group_width": 24,
        "default_size": 224,
        "block_type": "Y",
    },
    "y032": {
        "depths": [2, 5, 13, 1],
        "widths": [72, 216, 576, 1512],
        "group_width": 24,
        "default_size": 224,
        "block_type": "Y",
    },
    "y040": {
        "depths": [2, 6, 12, 2],
        "widths": [128, 192, 512, 1088],
        "group_width": 64,
        "default_size": 224,
        "block_type": "Y",
    },
    "y064": {
        "depths": [2, 7, 14, 2],
        "widths": [144, 288, 576, 1296],
        "group_width": 72,
        "default_size": 224,
        "block_type": "Y",
    },
    "y080": {
        "depths": [2, 4, 10, 1],
        "widths": [168, 448, 896, 2016],
        "group_width": 56,
        "default_size": 224,
        "block_type": "Y",
    },
    "y120": {
        "depths": [2, 5, 11, 1],
        "widths": [224, 448, 896, 2240],
        "group_width": 112,
        "default_size": 224,
        "block_type": "Y",
    },
    "y160": {
        "depths": [2, 4, 11, 1],
        "widths": [224, 448, 1232, 3024],
        "group_width": 112,
        "default_size": 224,
        "block_type": "Y",
    },
    "y320": {
        "depths": [2, 5, 12, 1],
        "widths": [232, 696, 1392, 3712],
        "group_width": 232,
        "default_size": 224,
        "block_type": "Y",
    },
}

BASE_DOCSTRING = """This class represents the {name} architecture.

  Reference:
    - [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)
    (CVPR 2020)

  For image classification use cases, see
  [this page for detailed examples](https://keras.io/api/applications/#usage-examples-for-image-classification-models).

  For transfer learning use cases, make sure to read the
  [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).


  The naming of models is as follows: `RegNet<block_type><flops>` where
  `block_type` is one of `(X, Y)` and `flops` signifies hundred million
  floating point operations. For example RegNetY064 corresponds to RegNet with
  Y block and 6.4 giga flops (64 hundred million flops).

  Args:
    include_rescaling: whether to rescale the inputs. If set to True,
        inputs will be passed through a `Rescaling(1/255.0)` layer.
    include_top: Whether to include the fully-connected
        layer at the top of the network.
    num_classes: Optional number of classes to classify images
        into, only to be specified if `include_top` is True.
    weights: One of `None` (random initialization), or the path to the weights
          file to be loaded, defaults to `None`.
    input_tensor: Optional Keras tensor (i.e. output of `layers.Input()`)
        to use as image input for the model.
    input_shape: Optional shape tuple, defaults to (None, None, 3).
        It should have exactly 3 inputs channels.
    pooling: Optional pooling mode for feature extraction
        when `include_top` is `False`, defaults to None.
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
        Defaults to `"softmax"`.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.

  Returns:
    A `keras.Model` instance.
"""  # noqa: E501


def apply_conv2d_bn(
    x,
    filters,
    kernel_size,
    strides=1,
    use_bias=False,
    groups=1,
    padding="valid",
    kernel_initializer="he_normal",
    batch_norm=True,
    activation="relu",
    name="",
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        groups=groups,
        use_bias=use_bias,
        padding=padding,
        kernel_initializer=kernel_initializer,
        name=name,
    )(x)

    if batch_norm:
        x = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name=name + "_bn"
        )(x)

    if activation is not None:
        x = layers.Activation(activation, name=name + f"_{activation}")(x)

    return x


def apply_stem(x, name=None):
    """Implementation of RegNet stem.

    (Common to all model variants)
    Args:
      x: Tensor, input tensor to the stem
      name: name prefix

    Returns:
      Output tensor of the Stem
    """
    if name is None:
        name = "stem" + str(backend.get_uid("stem"))

    x = apply_conv2d_bn(
        x=x,
        filters=32,
        kernel_size=(3, 3),
        strides=2,
        padding="same",
        name=name + "_stem_conv",
    )

    return x


def apply_x_block(
    inputs, filters_in, filters_out, group_width, stride=1, name=None
):
    """Implementation of X Block.
    References:
        - [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)

    Args:
      inputs: Tensor, input tensor to the block
      filters_in: int, filters in the input tensor
      filters_out: int, filters in the output tensor
      group_width: int, group width
      stride: int (or) tuple, stride of Conv layer
      name: str, name prefix
    Returns:
      Output tensor of the block
    """
    if name is None:
        name = str(backend.get_uid("xblock"))

    if filters_in != filters_out and stride == 1:
        raise ValueError(
            f"Input filters({filters_in}) and output "
            f"filters({filters_out}) "
            f"are not equal for stride {stride}. Input and output filters "
            f"must be equal for stride={stride}."
        )

    # Declare layers
    groups = filters_out // group_width

    if stride != 1:
        skip = apply_conv2d_bn(
            x=inputs,
            filters=filters_out,
            kernel_size=(1, 1),
            strides=stride,
            activation=None,
            name=name + "_skip_1x1",
        )
    else:
        skip = inputs

    # Build block
    # conv_1x1_1
    x = apply_conv2d_bn(
        x=inputs,
        filters=filters_out,
        kernel_size=(1, 1),
        name=name + "_conv_1x1_1",
    )

    # conv_3x3
    x = apply_conv2d_bn(
        x=x,
        filters=filters_out,
        kernel_size=(3, 3),
        strides=stride,
        groups=groups,
        padding="same",
        name=name + "_conv_3x3",
    )

    # conv_1x1_2
    x = apply_conv2d_bn(
        x=x,
        filters=filters_out,
        kernel_size=(1, 1),
        activation=None,
        name=name + "_conv_1x1_2",
    )

    x = layers.Activation("relu", name=name + "_exit_relu")(x + skip)

    return x


def apply_y_block(
    inputs,
    filters_in,
    filters_out,
    group_width,
    stride=1,
    squeeze_excite_ratio=0.25,
    name=None,
):
    """Implementation of Y Block.
    References:
        - [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)

    Args:
      inputs: Tensor, input tensor to the block
      filters_in: int, filters in the input tensor
      filters_out: int, filters in the output tensor
      group_width: int, group width
      stride: int (or) tuple, stride of Conv layer
      squeeze_excite_ratio: float, expansion ratio for Squeeze and Excite block
      name: str, name prefix
    Returns:
      Output tensor of the block
    """
    if name is None:
        name = str(backend.get_uid("yblock"))

    if filters_in != filters_out and stride == 1:
        raise ValueError(
            f"Input filters({filters_in}) and output "
            f"filters({filters_out}) "
            f"are not equal for stride {stride}. Input and output filters "
            f"must be equal for stride={stride}."
        )

    groups = filters_out // group_width

    if stride != 1:
        skip = apply_conv2d_bn(
            x=inputs,
            filters=filters_out,
            kernel_size=(1, 1),
            strides=stride,
            activation=None,
            name=name + "_skip_1x1",
        )
    else:
        skip = inputs

    # Build block
    # conv_1x1_1
    x = apply_conv2d_bn(
        x=inputs,
        filters=filters_out,
        kernel_size=(1, 1),
        name=name + "_conv_1x1_1",
    )

    # conv_3x3
    x = apply_conv2d_bn(
        x=x,
        filters=filters_out,
        kernel_size=(3, 3),
        strides=stride,
        groups=groups,
        padding="same",
        name=name + "_conv_3x3",
    )

    # Squeeze-Excitation block
    x = SqueezeAndExcite2D(
        filters_out,
        bottleneck_filters=filters_out * squeeze_excite_ratio,
        name=name,
    )(x)

    # conv_1x1_2
    x = apply_conv2d_bn(
        x=x,
        filters=filters_out,
        kernel_size=(1, 1),
        activation=None,
        name=name + "_conv_1x1_2",
    )

    x = layers.Activation("relu", name=name + "_exit_relu")(x + skip)

    return x


def apply_z_block(
    inputs,
    filters_in,
    filters_out,
    group_width,
    stride=1,
    squeeze_excite_ratio=0.25,
    bottleneck_ratio=0.25,
    name=None,
):
    """Implementation of Z block.

    References:
        - [Fast and Accurate Model Scaling](https://arxiv.org/abs/2103.06877).

    Args:
      inputs: Tensor, input tensor to the block
      filters_in: int, filters in the input tensor
      filters_out: int, filters in the output tensor
      group_width: int, group width
      stride: int (or) tuple, stride
      squeeze_excite_ratio: float, expansion ration for Squeeze and Excite block
      bottleneck_ratio: float, inverted bottleneck ratio
      name: str, name prefix
    Returns:
      Output tensor of the block
    """
    if name is None:
        name = str(backend.get_uid("zblock"))

    if filters_in != filters_out and stride == 1:
        raise ValueError(
            f"Input filters({filters_in}) and output filters({filters_out})"
            f"are not equal for stride {stride}. Input and output filters "
            f"must be equal for stride={stride}."
        )

    groups = filters_out // group_width

    inv_btlneck_filters = int(filters_out / bottleneck_ratio)

    # Build block
    # conv_1x1_1
    x = apply_conv2d_bn(
        x=inputs,
        filters=inv_btlneck_filters,
        kernel_size=(1, 1),
        name=name + "_conv_1x1_1",
        activation="silu",
    )

    # conv_3x3
    x = apply_conv2d_bn(
        x=x,
        filters=inv_btlneck_filters,
        kernel_size=(3, 3),
        strides=stride,
        groups=groups,
        padding="same",
        name=name + "_conv_3x3",
        activation="silu",
    )

    # Squeeze-Excitation block
    x = SqueezeAndExcite2D(
        inv_btlneck_filters,
        bottleneck_filter=inv_btlneck_filters * squeeze_excite_ratio,
        name=name,
    )(x)

    # conv_1x1_2
    x = apply_conv2d_bn(
        x=x,
        filters=filters_out,
        kernel_size=(1, 1),
        activation=None,
        name=name + "_conv_1x1_2",
    )

    if stride != 1:
        return x
    else:
        return x + inputs


def apply_stage(
    x, block_type, depth, group_width, filters_in, filters_out, name=None
):
    """Implementation of Stage in RegNet.

    Args:
      x: Tensor, input tensor to the stage
      block_type: must be one of "X", "Y", "Z"
      depth: int, depth of stage, number of blocks to use
      group_width: int, group width of all blocks in  this stage
      filters_in: int, input filters to this stage
      filters_out: int, output filters from this stage
      name: str, name prefix

    Returns:
      Output tensor of the block
    """
    if name is None:
        name = str(backend.get_uid("stage"))

    if block_type == "X":
        x = apply_x_block(
            x,
            filters_in,
            filters_out,
            group_width,
            stride=2,
            name=f"{name}_XBlock_0",
        )
        for i in range(1, depth):
            x = apply_x_block(
                x,
                filters_out,
                filters_out,
                group_width,
                name=f"{name}_XBlock_{i}",
            )
    elif block_type == "Y":
        x = apply_y_block(
            x,
            filters_in,
            filters_out,
            group_width,
            stride=2,
            name=name + "_YBlock_0",
        )
        for i in range(1, depth):
            x = apply_y_block(
                x,
                filters_out,
                filters_out,
                group_width,
                name=f"{name}_YBlock_{i}",
            )
    elif block_type == "Z":
        x = apply_z_block(
            x,
            filters_in,
            filters_out,
            group_width,
            stride=2,
            name=f"{name}_ZBlock_0",
        )
        for i in range(1, depth):
            x = apply_z_block(
                x,
                filters_out,
                filters_out,
                group_width,
                name=f"{name}_ZBlock_{i}",
            )
    else:
        raise NotImplementedError(
            f"Block type `{block_type}` not recognized."
            f"block_type must be one of (`X`, `Y`, `Z`). "
        )
    return x


def apply_head(x, num_classes=None, name=None, activation=None):
    """Implementation of classification head of RegNet.

    Args:
      x: Tensor, input to the head block
      num_classes: int, number of classes for Dense layer
      name: str, name prefix

    Returns:
      Output logits tensor.
    """
    if name is None:
        name = str(backend.get_uid("head"))

    x = layers.GlobalAveragePooling2D(name=name + "_head_gap")(x)
    x = layers.Dense(
        num_classes, name=name + "head_dense", activation=activation
    )(x)
    return x


@keras.utils.register_keras_serializable(package="keras_cv.models")
class RegNet(keras.Model):
    """
    This class represents the architecture of RegNet
    Args:
        depths: iterable, Contains depths for each individual stages.
        widths: iterable, Contains output channel width of each individual
            stages
        group_width: int, Number of channels to be used in each group. See
            grouped convolutions for more information.
        block_type: Must be one of `{"X", "Y", "Z"}`. For more details see the
            papers "Designing network design spaces" and "Fast and Accurate
            Model Scaling"
        default_size: tuple (or) list, default input image size.
        model_name: str, An optional name for the model.
        include_rescaling: bool, whether to rescale the inputs. If set to
            True, inputs will be passed through a `Rescaling(1/255.0)` layer.
        include_top: bool, Whether to include the fully-connected
            layer at the top of the network.
        num_classes: int, Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        weights: str, One of `None` (random initialization), or the path to the
            weights file to be loaded, defaults to `None`.
        input_tensor: Tensor, Optional Keras tensor (i.e. output of
            `layers.Input()`) to use as image input for the model.
        input_shape: Optional shape tuple, defaults to (None, None, 3).
            It should have exactly 3 inputs channels.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`, defaults to None.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classifier_activation: A `str` or callable. The activation function to
            use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top"
            layer. Defaults to `"softmax"`.
    """

    def __init__(
        self,
        depths,
        widths,
        group_width,
        block_type,
        include_rescaling,
        include_top,
        num_classes=None,
        model_name="regnet",
        weights=None,
        input_tensor=None,
        input_shape=(None, None, 3),
        pooling=None,
        classifier_activation="softmax",
        **kwargs,
    ):
        if not (weights is None or tf.io.gfile.exists(weights)):
            raise ValueError(
                "The `weights` argument should be either "
                "`None` (random initialization) "
                "or the path to the weights file to be loaded."
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

        img_input = utils.parse_model_inputs(input_shape, input_tensor)
        x = img_input

        if include_rescaling:
            x = layers.Rescaling(scale=1.0 / 255.0)(x)
        x = apply_stem(x, name=model_name)

        in_channels = x.shape[-1]  # Output from Stem

        NUM_STAGES = 4

        for stage_index in range(NUM_STAGES):
            depth = depths[stage_index]
            out_channels = widths[stage_index]

            x = apply_stage(
                x,
                block_type,
                depth,
                group_width,
                in_channels,
                out_channels,
                name=model_name + "_Stage_" + str(stage_index),
            )
            in_channels = out_channels

        if include_top:
            x = apply_head(
                x, num_classes=num_classes, activation=classifier_activation
            )
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling2D()(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D()(x)

        super().__init__(inputs=img_input, outputs=x, name=model_name, **kwargs)

        # Load weights.
        if weights is not None:
            self.load_weights(weights)

        self.depths = depths
        self.widths = widths
        self.group_width = group_width
        self.block_type = block_type
        self.include_rescaling = include_rescaling
        self.include_top = include_top
        self.num_classes = num_classes
        self.model_name = model_name
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.classifier_activation = classifier_activation

    def get_config(self):
        return {
            "depths": self.depths,
            "widths": self.widths,
            "group_width": self.group_width,
            "block_type": self.block_type,
            "include_rescaling": self.include_rescaling,
            "include_top": self.include_top,
            "num_classes": self.num_classes,
            "model_name": self.model_name,
            "input_tensor": self.input_tensor,
            "input_shape": self.input_shape[1:],
            "pooling": self.pooling,
            "classifier_activation": self.classifier_activation,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Instantiating variants
def RegNetX002(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnetx002",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["x002"]["depths"],
        MODEL_CONFIGS["x002"]["widths"],
        MODEL_CONFIGS["x002"]["group_width"],
        MODEL_CONFIGS["x002"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnetx002"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def RegNetX004(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnetx004",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["x004"]["depths"],
        MODEL_CONFIGS["x004"]["widths"],
        MODEL_CONFIGS["x004"]["group_width"],
        MODEL_CONFIGS["x004"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnetx004"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def RegNetX006(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnetx006",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["x006"]["depths"],
        MODEL_CONFIGS["x006"]["widths"],
        MODEL_CONFIGS["x006"]["group_width"],
        MODEL_CONFIGS["x006"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnetx006"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def RegNetX008(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnetx008",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["x008"]["depths"],
        MODEL_CONFIGS["x008"]["widths"],
        MODEL_CONFIGS["x008"]["group_width"],
        MODEL_CONFIGS["x008"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnetx008"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def RegNetX016(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnetx016",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["x016"]["depths"],
        MODEL_CONFIGS["x016"]["widths"],
        MODEL_CONFIGS["x016"]["group_width"],
        MODEL_CONFIGS["x016"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnetx016"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def RegNetX032(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnetx032",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["x032"]["depths"],
        MODEL_CONFIGS["x032"]["widths"],
        MODEL_CONFIGS["x032"]["group_width"],
        MODEL_CONFIGS["x032"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnetx032"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def RegNetX040(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnetx040",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["x040"]["depths"],
        MODEL_CONFIGS["x040"]["widths"],
        MODEL_CONFIGS["x040"]["group_width"],
        MODEL_CONFIGS["x040"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnetx040"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def RegNetX064(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnetx064",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["x064"]["depths"],
        MODEL_CONFIGS["x064"]["widths"],
        MODEL_CONFIGS["x064"]["group_width"],
        MODEL_CONFIGS["x064"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnetx064"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def RegNetX080(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnetx080",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["x080"]["depths"],
        MODEL_CONFIGS["x080"]["widths"],
        MODEL_CONFIGS["x080"]["group_width"],
        MODEL_CONFIGS["x080"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnetx080"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def RegNetX120(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnetx120",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["x120"]["depths"],
        MODEL_CONFIGS["x120"]["widths"],
        MODEL_CONFIGS["x120"]["group_width"],
        MODEL_CONFIGS["x120"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnetx120"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def RegNetX160(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnetx160",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["x160"]["depths"],
        MODEL_CONFIGS["x160"]["widths"],
        MODEL_CONFIGS["x160"]["group_width"],
        MODEL_CONFIGS["x160"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnetx160"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def RegNetX320(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnetx320",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["x320"]["depths"],
        MODEL_CONFIGS["x320"]["widths"],
        MODEL_CONFIGS["x320"]["group_width"],
        MODEL_CONFIGS["x320"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnetx320"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def RegNetY002(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnety002",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["y002"]["depths"],
        MODEL_CONFIGS["y002"]["widths"],
        MODEL_CONFIGS["y002"]["group_width"],
        MODEL_CONFIGS["y002"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnety002"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def RegNetY004(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnety004",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["y004"]["depths"],
        MODEL_CONFIGS["y004"]["widths"],
        MODEL_CONFIGS["y004"]["group_width"],
        MODEL_CONFIGS["y004"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnety004"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def RegNetY006(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnety006",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["y006"]["depths"],
        MODEL_CONFIGS["y006"]["widths"],
        MODEL_CONFIGS["y006"]["group_width"],
        MODEL_CONFIGS["y006"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnety006"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def RegNetY008(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnety008",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["y008"]["depths"],
        MODEL_CONFIGS["y008"]["widths"],
        MODEL_CONFIGS["y008"]["group_width"],
        MODEL_CONFIGS["y008"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnety008"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def RegNetY016(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnety016",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["y016"]["depths"],
        MODEL_CONFIGS["y016"]["widths"],
        MODEL_CONFIGS["y016"]["group_width"],
        MODEL_CONFIGS["y016"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnety016"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def RegNetY032(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnety032",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["y032"]["depths"],
        MODEL_CONFIGS["y032"]["widths"],
        MODEL_CONFIGS["y032"]["group_width"],
        MODEL_CONFIGS["y032"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnety032"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def RegNetY040(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnety040",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["y040"]["depths"],
        MODEL_CONFIGS["y040"]["widths"],
        MODEL_CONFIGS["y040"]["group_width"],
        MODEL_CONFIGS["y040"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnety040"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def RegNetY064(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnety064",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["y064"]["depths"],
        MODEL_CONFIGS["y064"]["widths"],
        MODEL_CONFIGS["y064"]["group_width"],
        MODEL_CONFIGS["y064"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnety064"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def RegNetY080(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnety080",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["y080"]["depths"],
        MODEL_CONFIGS["y080"]["widths"],
        MODEL_CONFIGS["y080"]["group_width"],
        MODEL_CONFIGS["y080"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnety080"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def RegNetY120(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnety120",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["y120"]["depths"],
        MODEL_CONFIGS["y120"]["widths"],
        MODEL_CONFIGS["y120"]["group_width"],
        MODEL_CONFIGS["y120"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnety120"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def RegNetY160(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnety160",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["y160"]["depths"],
        MODEL_CONFIGS["y160"]["widths"],
        MODEL_CONFIGS["y160"]["group_width"],
        MODEL_CONFIGS["y160"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnety160"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def RegNetY320(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_tensor=None,
    input_shape=(None, None, 3),
    pooling=None,
    model_name="regnety320",
    classifier_activation="softmax",
    **kwargs,
):
    return RegNet(
        MODEL_CONFIGS["y320"]["depths"],
        MODEL_CONFIGS["y320"]["widths"],
        MODEL_CONFIGS["y320"]["group_width"],
        MODEL_CONFIGS["y320"]["block_type"],
        model_name=model_name,
        include_top=include_top,
        include_rescaling=include_rescaling,
        weights=parse_weights(weights, include_top, "regnety320"),
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


RegNetX002.__doc__ = BASE_DOCSTRING.format(name="RegNetX002")
RegNetX004.__doc__ = BASE_DOCSTRING.format(name="RegNetX004")
RegNetX006.__doc__ = BASE_DOCSTRING.format(name="RegNetX006")
RegNetX008.__doc__ = BASE_DOCSTRING.format(name="RegNetX008")
RegNetX016.__doc__ = BASE_DOCSTRING.format(name="RegNetX016")
RegNetX032.__doc__ = BASE_DOCSTRING.format(name="RegNetX032")
RegNetX040.__doc__ = BASE_DOCSTRING.format(name="RegNetX040")
RegNetX064.__doc__ = BASE_DOCSTRING.format(name="RegNetX064")
RegNetX080.__doc__ = BASE_DOCSTRING.format(name="RegNetX080")
RegNetX120.__doc__ = BASE_DOCSTRING.format(name="RegNetX120")
RegNetX160.__doc__ = BASE_DOCSTRING.format(name="RegNetX160")
RegNetX320.__doc__ = BASE_DOCSTRING.format(name="RegNetX320")

RegNetY002.__doc__ = BASE_DOCSTRING.format(name="RegNetY002")
RegNetY004.__doc__ = BASE_DOCSTRING.format(name="RegNetY004")
RegNetY006.__doc__ = BASE_DOCSTRING.format(name="RegNetY006")
RegNetY008.__doc__ = BASE_DOCSTRING.format(name="RegNetY008")
RegNetY016.__doc__ = BASE_DOCSTRING.format(name="RegNetY016")
RegNetY032.__doc__ = BASE_DOCSTRING.format(name="RegNetY032")
RegNetY040.__doc__ = BASE_DOCSTRING.format(name="RegNetY040")
RegNetY064.__doc__ = BASE_DOCSTRING.format(name="RegNetY064")
RegNetY080.__doc__ = BASE_DOCSTRING.format(name="RegNetY080")
RegNetY120.__doc__ = BASE_DOCSTRING.format(name="RegNetY120")
RegNetY160.__doc__ = BASE_DOCSTRING.format(name="RegNetY160")
RegNetY320.__doc__ = BASE_DOCSTRING.format(name="RegNetY320")
