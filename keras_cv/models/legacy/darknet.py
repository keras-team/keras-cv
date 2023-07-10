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

"""DarkNet models for KerasCV.
Reference:
    - [YoloV3 Paper](https://arxiv.org/abs/1804.02767)
    - [YoloV3 implementation](https://github.com/ultralytics/yolov3)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers

from keras_cv.models.legacy import utils
from keras_cv.models.legacy.weights import parse_weights


def DarknetConvBlock(
    filters, kernel_size, strides, use_bias=False, activation="silu", name=None
):
    """The basic conv block used in Darknet. Applies Conv2D followed by a
    BatchNorm.

    Args:
        filters: Integer, the dimensionality of the output space (i.e. the
            number of output filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            height and width of the 2D convolution window. Can be a single
            integer to specify the same value both dimensions.
        strides: An integer or tuple/list of 2 integers, specifying the strides
            of the convolution along the height and width. Can be a single
            integer to the same value both dimensions.
        use_bias: Boolean, whether the layer uses a bias vector.
        activation: the activation applied after the BatchNorm layer. One of
            "silu", "relu" or "leaky_relu", defaults to "silu".
        name: the prefix for the layer names used in the block.
    """

    if name is None:
        name = f"conv_block{backend.get_uid('conv_block')}"

    model_layers = [
        layers.Conv2D(
            filters,
            kernel_size,
            strides,
            padding="same",
            use_bias=use_bias,
            name=name + "_conv",
        ),
        layers.BatchNormalization(name=name + "_bn"),
    ]

    if activation == "silu":
        model_layers.append(layers.Lambda(lambda x: keras.activations.swish(x)))
    elif activation == "relu":
        model_layers.append(layers.ReLU())
    elif activation == "leaky_relu":
        model_layers.append(layers.LeakyReLU(0.1))

    return keras.Sequential(model_layers, name=None)


def ResidualBlocks(filters, num_blocks, name=None):
    """A residual block used in DarkNet models, repeated `num_blocks` times.

    Args:
        filters: Integer, the dimensionality of the output spaces (i.e. the
            number of output filters in used the blocks).
        num_blocks: number of times the residual connections are repeated
        name: the prefix for the layer names used in the block.

    Returns:
        a function that takes an input Tensor representing a ResidualBlock.
    """

    if name is None:
        name = f"residual_block{backend.get_uid('residual_block')}"

    def apply(x):
        x = DarknetConvBlock(
            filters,
            kernel_size=3,
            strides=2,
            activation="leaky_relu",
            name=f"{name}_conv1",
        )(x)

        for i in range(1, num_blocks + 1):
            residual = x

            x = DarknetConvBlock(
                filters // 2,
                kernel_size=1,
                strides=1,
                activation="leaky_relu",
                name=f"{name}_conv{2*i}",
            )(x)
            x = DarknetConvBlock(
                filters,
                kernel_size=3,
                strides=1,
                activation="leaky_relu",
                name=f"{name}_conv{2*i + 1}",
            )(x)

            if i == num_blocks:
                x = layers.Add(name=f"{name}_out")([residual, x])
            else:
                x = layers.Add(name=f"{name}_add_{i}")([residual, x])

        return x

    return apply


def SpatialPyramidPoolingBottleneck(
    filters,
    hidden_filters=None,
    kernel_sizes=(5, 9, 13),
    activation="silu",
    name=None,
):
    """Spatial pyramid pooling layer used in YOLOv3-SPP

    Args:
        filters: Integer, the dimensionality of the output spaces (i.e. the
            number of output filters in used the blocks).
        hidden_filters: Integer, the dimensionality of the intermediate
            bottleneck space (i.e. the number of output filters in the
            bottleneck convolution). If None, it will be equal to filters.
            Defaults to None.
        kernel_sizes: A list or tuple representing all the pool sizes used for
            the pooling layers, defaults to (5, 9, 13).
        activation: Activation for the conv layers, defaults to "silu".
        name: the prefix for the layer names used in the block.

    Returns:
        a function that takes an input Tensor representing an
        SpatialPyramidPoolingBottleneck.
    """
    if name is None:
        name = f"spp{backend.get_uid('spp')}"

    if hidden_filters is None:
        hidden_filters = filters

    def apply(x):
        x = DarknetConvBlock(
            hidden_filters,
            kernel_size=1,
            strides=1,
            activation=activation,
            name=f"{name}_conv1",
        )(x)
        x = [x]

        for kernel_size in kernel_sizes:
            x.append(
                layers.MaxPooling2D(
                    kernel_size,
                    strides=1,
                    padding="same",
                    name=f"{name}_maxpool_{kernel_size}",
                )(x[0])
            )

        x = layers.Concatenate(name=f"{name}_concat")(x)
        x = DarknetConvBlock(
            filters,
            kernel_size=1,
            strides=1,
            activation=activation,
            name=f"{name}_conv2",
        )(x)

        return x

    return apply


BASE_DOCSTRING = """Represents the {name} architecture.

    Although the {name} architecture is commonly used for detection tasks, it is
    possible to extract the intermediate dark2 to dark5 layers from the model
    for creating a feature pyramid Network.

    Reference:
        - [YoloV3 Paper](https://arxiv.org/abs/1804.02767)
        - [YoloV3 implementation](https://github.com/ultralytics/yolov3)

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        include_rescaling: bool, whether to rescale the inputs. If set to
            True, inputs will be passed through a `Rescaling(1/255.0)` layer.
        include_top: bool, whether to include the fully-connected layer at the
            top of the network. If provided, `num_classes` must be provided.
        num_classes: integer, optional number of classes to classify images
            into. Only to be specified if `include_top` is True.
        weights: one of `None` (random initialization), or a pretrained weight
            file path.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e., output of `layers.Input()`)
            to use as image input for the model.
        pooling: optional pooling mode for feature extraction when `include_top`
            is `False`.
            - `None` means that the output of the model will be the 4D tensor
                output of the last convolutional block.
            - `avg` means that global average pooling will be applied to the
                output of the last convolutional block, and thus the output of
                the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        name: string, optional name to pass to the model, defaults to "{name}".

    Returns:
        A `keras.Model` instance.
"""  # noqa: E501


@keras.utils.register_keras_serializable(package="keras_cv.models")
class DarkNet(keras.Model):

    """Represents the DarkNet architecture.

    The DarkNet architecture is commonly used for detection tasks. It is
    possible to extract the intermediate dark2 to dark5 layers from the model
    for creating a feature pyramid Network.

    Reference:
        - [YoloV3 Paper](https://arxiv.org/abs/1804.02767)
        - [YoloV3 implementation](https://github.com/ultralytics/yolov3)
    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        blocks: integer, numbers of building blocks from the layer dark2 to
            layer dark5.
        include_rescaling: bool, whether to rescale the inputs. If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        include_top: bool, whether to include the fully-connected layer at the
            top of the network. If provided, `num_classes` must be provided.
        num_classes: integer, optional number of classes to classify images
            into. Only to be specified if `include_top` is True.
        weights: one of `None` (random initialization) or a pretrained weight
            file path.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e., output of `layers.Input()`)
            to use as image input for the model.
        pooling: optional pooling mode for feature extraction when `include_top`
            is `False`.
            - `None` means that the output of the model will be the 4D tensor
                output of the last convolutional block.
            - `avg` means that global average pooling will be applied to the
                output of the last convolutional block, and thus the output of
                the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        classifier_activation: A `str` or callable. The activation function to
            use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top"
            layer.
        name: string, optional name to pass to the model, defaults to "DarkNet".

    Returns:
        A `keras.Model` instance.
    """  # noqa: E501

    def __init__(
        self,
        blocks,
        include_rescaling,
        include_top,
        num_classes=None,
        weights=None,
        input_shape=(None, None, 3),
        input_tensor=None,
        pooling=None,
        classifier_activation="softmax",
        name="DarkNet",
        **kwargs,
    ):
        if weights and not tf.io.gfile.exists(weights):
            raise ValueError(
                "The `weights` argument should be either `None` or the path to "
                "the weights file to be loaded. Weights file not found at "
                f"location: {weights}"
            )

        if include_top and not num_classes:
            raise ValueError(
                "If `include_top` is True, you should specify `num_classes`. "
                f"Received: num_classes={num_classes}"
            )

        inputs = utils.parse_model_inputs(input_shape, input_tensor)

        x = inputs
        if include_rescaling:
            x = layers.Rescaling(1 / 255.0)(x)

        # stem
        x = DarknetConvBlock(
            filters=32,
            kernel_size=3,
            strides=1,
            activation="leaky_relu",
            name="stem_conv",
        )(x)
        x = ResidualBlocks(
            filters=64, num_blocks=1, name="stem_residual_block"
        )(x)

        # filters for the ResidualBlock outputs
        filters = [128, 256, 512, 1024]

        # layer_num is used for naming the residual blocks
        # (starts with dark2, hence 2)
        layer_num = 2

        for filter, block in zip(filters, blocks):
            x = ResidualBlocks(
                filters=filter,
                num_blocks=block,
                name=f"dark{layer_num}_residual_block",
            )(x)
            layer_num += 1

        # remaining dark5 layers
        x = DarknetConvBlock(
            filters=512,
            kernel_size=1,
            strides=1,
            activation="leaky_relu",
            name="dark5_conv1",
        )(x)
        x = DarknetConvBlock(
            filters=1024,
            kernel_size=3,
            strides=1,
            activation="leaky_relu",
            name="dark5_conv2",
        )(x)
        x = SpatialPyramidPoolingBottleneck(
            512, activation="leaky_relu", name="dark5_spp"
        )(x)
        x = DarknetConvBlock(
            filters=1024,
            kernel_size=3,
            strides=1,
            activation="leaky_relu",
            name="dark5_conv3",
        )(x)
        x = DarknetConvBlock(
            filters=512,
            kernel_size=1,
            strides=1,
            activation="leaky_relu",
            name="dark5_conv4",
        )(x)

        if include_top:
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            x = layers.Dense(
                num_classes,
                activation=classifier_activation,
                name="predictions",
            )(x)
        elif pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D(name="max_pool")(x)

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        if weights is not None:
            self.load_weights(weights)

        self.blocks = blocks
        self.include_rescaling = include_rescaling
        self.include_top = include_top
        self.num_classes = num_classes
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.classifier_activation = classifier_activation

    def get_config(self):
        return {
            "blocks": self.blocks,
            "include_rescaling": self.include_rescaling,
            "include_top": self.include_top,
            "num_classes": self.num_classes,
            "input_shape": self.input_shape[1:],
            "input_tensor": self.input_tensor,
            "pooling": self.pooling,
            "classifier_activation": self.classifier_activation,
            "name": self.name,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def DarkNet21(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    name="DarkNet21",
    **kwargs,
):
    return DarkNet(
        [1, 2, 2, 1],
        include_rescaling=include_rescaling,
        include_top=include_top,
        num_classes=num_classes,
        weights=parse_weights(weights, include_top, "darknet"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        name=name,
        **kwargs,
    )


def DarkNet53(
    *,
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    name="DarkNet53",
    **kwargs,
):
    return DarkNet(
        [2, 8, 8, 4],
        include_rescaling=include_rescaling,
        include_top=include_top,
        num_classes=num_classes,
        weights=parse_weights(weights, include_top, "darknet53"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        name=name,
        **kwargs,
    )


setattr(DarkNet21, "__doc__", BASE_DOCSTRING.format(name="DarkNet21"))
setattr(DarkNet53, "__doc__", BASE_DOCSTRING.format(name="DarkNet53"))
