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


"""ConvMixer models for Keras.

References:
- [Patches Are All You Need?](https://arxiv.org/abs/2201.09792)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras_cv.src.models.legacy import utils
from keras_cv.src.models.legacy.weights import parse_weights

MODEL_CONFIGS = {
    "ConvMixer_1536_20": {
        "dim": 1536,
        "depth": 20,
        "patch_size": 7,
        "kernel_size": 9,
    },
    "ConvMixer_1536_24": {
        "dim": 1536,
        "depth": 24,
        "patch_size": 14,
        "kernel_size": 9,
    },
    "ConvMixer_768_32": {
        "dim": 768,
        "depth": 32,
        "patch_size": 7,
        "kernel_size": 7,
    },
    "ConvMixer_1024_16": {
        "dim": 1024,
        "depth": 16,
        "patch_size": 7,
        "kernel_size": 9,
    },
    "ConvMixer_512_16": {
        "dim": 512,
        "depth": 16,
        "patch_size": 7,
        "kernel_size": 8,
    },
}

BASE_DOCSTRING = """Instantiates the {name} architecture.

    Reference:
        - [Patches Are All You Need?](https://arxiv.org/abs/2201.09792)

    This class represents a Keras {name} model.

    For transfer learning use cases, make sure to read the [guide to transfer
        learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        include_rescaling: bool, whether to rescale the inputs. If set to
            True, inputs will be passed through a `Rescaling(1/255.0)` layer.
        include_top: bool, whether to include the fully-connected layer at the
            top of the network. If provided, num_classes must be provided.
        num_classes: integer, optional number of classes to classify images
            into. Only to be specified if `include_top` is True.
        weights: one of `None` (random initialization), a pretrained weight file
            path, or a reference to pre-trained weights (e.g.
            'imagenet/classification')(see available pre-trained weights in
            weights.py)
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e., output of `layers.Input()`)
            to use as image input for the model.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be the 4D tensor
                output of the last convolutional block.
            - `avg` means that global average pooling will be applied to the
                output of the last convolutional block, and thus the output of
                the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        name: string, optional name to pass to the model, defaults to "{name}".

    Returns:
      A `keras.Model` instance.
"""


def apply_conv_mixer_layer(x, dim, kernel_size):
    """ConvMixerLayer module.
    Args:
        x: input tensor.
        dim: integer, filters of the layer in a block.
        kernel_size: integer, kernel size of the Conv2D layers.
    Returns:
        the updated input tensor.
    """

    residual = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = tf.nn.gelu(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])

    x = layers.Conv2D(dim, kernel_size=1)(x)
    x = tf.nn.gelu(x)
    x = layers.BatchNormalization()(x)
    return x


def apply_patch_embed(x, dim, patch_size):
    """Implementation for Extracting Patch Embeddings.
    Args:
        x: input tensor.
        dim: integer, filters of the layer in a block.
        patch_size: integer, Size of patches.
    Returns:
        the updated input tensor.
    """

    x = layers.Conv2D(filters=dim, kernel_size=patch_size, strides=patch_size)(
        x
    )
    x = tf.nn.gelu(x)
    x = layers.BatchNormalization()(x)
    return x


@keras.utils.register_keras_serializable(package="keras_cv.models")
class ConvMixer(keras.Model):
    """Instantiates the ConvMixer architecture.

    Args:
        dim: integer, number of filters.
        depth: integer, number of ConvMixer Layer.
        patch_size: integer, size of the patches.
        kernel_size: integer, kernel size for Conv2D layers.
        include_top: bool, whether to include the fully-connected layer at the
            top of the network.
        include_rescaling: bool, whether to rescale the inputs. If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        name: string, optional name to pass to the model, defaults to
            "ConvMixer".
        weights: one of `None` (random initialization) or the path to the
            weights file to be loaded.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e., output of `layers.Input()`)
            to use as image input for the model.
        pooling: optional pooling mode for feature extraction when `include_top`
            is `False`.
            - `None` means that the output of the model will be the 4D tensor
                output of the last convolutional layer.
            - `avg` means that global average pooling will be applied to the
                output of the last convolutional layer, and thus the output of
                the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        num_classes: integer, optional number of classes to classify images
            into. Only to be specified if `include_top` is True.
        classifier_activation: A `str` or callable. The activation function to
            use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top"
            layer.
        **kwargs: Pass-through keyword arguments to `keras.Model`.

    Returns:
      A `keras.Model` instance.
    """

    def __init__(
        self,
        dim,
        depth,
        patch_size,
        kernel_size,
        include_top,
        include_rescaling,
        name="ConvMixer",
        weights=None,
        input_shape=(None, None, 3),
        input_tensor=None,
        pooling=None,
        num_classes=None,
        classifier_activation="softmax",
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
                "If `include_top` is True, you should specify `classes`. "
                f"Received: classes={num_classes}"
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
        x = apply_patch_embed(x, dim, patch_size)

        for _ in range(depth):
            x = apply_conv_mixer_layer(x, dim, kernel_size)

        if include_top:
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            x = layers.Dense(
                num_classes,
                activation=classifier_activation,
                name="predictions",
            )(x)
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D(name="max_pool")(x)

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)

        if weights is not None:
            self.load_weights(weights)

        self.dim = dim
        self.depth = depth
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.include_top = include_top
        self.include_rescaling = include_rescaling
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def get_config(self):
        return {
            "dim": self.dim,
            "depth": self.depth,
            "patch_size": self.patch_size,
            "kernel_size": self.kernel_size,
            "include_top": self.include_top,
            "include_rescaling": self.include_rescaling,
            "name": self.name,
            "input_shape": self.input_shape[1:],
            "input_tensor": self.input_tensor,
            "pooling": self.pooling,
            "num_classes": self.num_classes,
            "classifier_activation": self.classifier_activation,
            "trainable": self.trainable,
        }


def ConvMixer_1536_20(
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="ConvMixer_1536_20",
    **kwargs,
):
    return ConvMixer(
        dim=MODEL_CONFIGS["ConvMixer_1536_20"]["dim"],
        depth=MODEL_CONFIGS["ConvMixer_1536_20"]["depth"],
        patch_size=MODEL_CONFIGS["ConvMixer_1536_20"]["patch_size"],
        kernel_size=MODEL_CONFIGS["ConvMixer_1536_20"]["kernel_size"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ConvMixer_1536_24(
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="ConvMixer_1536_24",
    **kwargs,
):
    return ConvMixer(
        dim=MODEL_CONFIGS["ConvMixer_1536_24"]["dim"],
        depth=MODEL_CONFIGS["ConvMixer_1536_24"]["depth"],
        patch_size=MODEL_CONFIGS["ConvMixer_1536_24"]["patch_size"],
        kernel_size=MODEL_CONFIGS["ConvMixer_1536_24"]["kernel_size"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ConvMixer_768_32(
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="ConvMixer_768_32",
    **kwargs,
):
    return ConvMixer(
        dim=MODEL_CONFIGS["ConvMixer_768_32"]["dim"],
        depth=MODEL_CONFIGS["ConvMixer_768_32"]["depth"],
        patch_size=MODEL_CONFIGS["ConvMixer_768_32"]["patch_size"],
        kernel_size=MODEL_CONFIGS["ConvMixer_768_32"]["kernel_size"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ConvMixer_1024_16(
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="ConvMixer_1024_16",
    **kwargs,
):
    return ConvMixer(
        dim=MODEL_CONFIGS["ConvMixer_1024_16"]["dim"],
        depth=MODEL_CONFIGS["ConvMixer_1024_16"]["depth"],
        patch_size=MODEL_CONFIGS["ConvMixer_1024_16"]["patch_size"],
        kernel_size=MODEL_CONFIGS["ConvMixer_1024_16"]["kernel_size"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ConvMixer_512_16(
    include_rescaling,
    include_top,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="ConvMixer_512_16",
    **kwargs,
):
    return ConvMixer(
        dim=MODEL_CONFIGS["ConvMixer_512_16"]["dim"],
        depth=MODEL_CONFIGS["ConvMixer_512_16"]["depth"],
        patch_size=MODEL_CONFIGS["ConvMixer_512_16"]["patch_size"],
        kernel_size=MODEL_CONFIGS["ConvMixer_512_16"]["kernel_size"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        name=name,
        weights=parse_weights(weights, include_top, "convmixer_512_16"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        classifier_activation=classifier_activation,
        **kwargs,
    )


setattr(
    ConvMixer_1536_20,
    "__doc__",
    BASE_DOCSTRING.format(name="ConvMixer_1536_20"),
)
setattr(
    ConvMixer_1536_24,
    "__doc__",
    BASE_DOCSTRING.format(name="ConvMixer_1536_24"),
)
setattr(
    ConvMixer_768_32, "__doc__", BASE_DOCSTRING.format(name="ConvMixer_768_32")
)
setattr(
    ConvMixer_1024_16,
    "__doc__",
    BASE_DOCSTRING.format(name="ConvMixer_1024_16"),
)
setattr(
    ConvMixer_512_16, "__doc__", BASE_DOCSTRING.format(name="ConvMixer_512_16")
)
