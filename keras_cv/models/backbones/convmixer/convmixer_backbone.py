# Copyright 2023 The KerasCV Authors
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

import copy

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras_cv.models import utils
from keras_cv.models.backbones.backbone import Backbone
from keras_cv.models.backbones.convmixer.convmixer_backbone_presets import (
    backbone_presets,
)
from keras_cv.models.backbones.convmixer.convmixer_backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.utils.python_utils import classproperty

BASE_DOCSTRING = """Instantiates the {name} architecture.

    Reference:
        - [Patches Are All You Need?](https://arxiv.org/abs/2201.09792)

    This class represents a Keras {name} model.

    For transfer learning use cases, make sure to read the [guide to transfer
        learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        include_rescaling: bool, whether or not to rescale the inputs. If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        include_top: bool, whether to include the fully-connected layer at the top of the
            network. If provided, num_classes must be provided.
        num_classes: integer, optional number of classes to classify images into. Only to be
            specified if `include_top` is True.
        weights: one of `None` (random initialization), a pretrained weight file
            path, or a reference to pre-trained weights (e.g. 'imagenet/classification')
            (see available pre-trained weights in weights.py)
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e., output of `layers.Input()`)
            to use as image input for the model.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be the 4D tensor output
                of the last convolutional block.
            - `avg` means that global average pooling will be applied to the output
                of the last convolutional block, and thus the output of the model will
                be a 2D tensor.
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
class ConvMixerBackbone(Backbone):
    """Instantiates the ConvMixer architecture.

    Args:
        dim: integer, number of filters.
        depth: integer, number of ConvMixer Layer.
        patch_size: integer, size of the patches.
        kernel_size: integer, kernel size for Conv2D layers.
        include_top: bool, whether to include the fully-connected
            layer at the top of the network.
        include_rescaling: bool, whether to rescale the inputs. If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        name: string, optional name to pass to the model, defaults to "ConvMixer".
        weights: one of `None` (random initialization)
            or the path to the weights file to be loaded.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e., output of `layers.Input()`)
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
        num_classes: integer, optional number of classes to classify images
            into. Only to be specified if `include_top` is True.
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.
        **kwargs: Pass-through keyword arguments to `keras.Model`.

    Returns:
      A `keras.Model` instance.
    """

    def __init__(
        self,
        *,
        dim,
        depth,
        patch_size,
        kernel_size,
        include_rescaling,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        inputs = utils.parse_model_inputs(input_shape, input_tensor)
        x = inputs

        if include_rescaling:
            x = layers.Rescaling(1 / 255.0)(x)
        x = apply_patch_embed(x, dim, patch_size)

        for _ in range(depth):
            x = apply_conv_mixer_layer(x, dim, kernel_size)

        super().__init__(inputs=inputs, outputs=x, **kwargs)

        self.dim = dim
        self.depth = depth
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.include_rescaling = include_rescaling
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "depth": self.depth,
                "patch_size": self.patch_size,
                "kernel_size": self.kernel_size,
                "include_rescaling": self.include_rescaling,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
            }
        )
        return config

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy(backbone_presets)

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include weights."""
        return copy.deepcopy(backbone_presets_with_weights)


class ConvMixer_1536_20Backbone(ConvMixerBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ConvMixerBackbone.from_preset("convmixer_1536_20", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}


class ConvMixer_1536_24Backbone(ConvMixerBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ConvMixerBackbone.from_preset("convmixer_1536_24", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}


class ConvMixer_768_32Backbone(ConvMixerBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ConvMixerBackbone.from_preset("convmixer_768_32", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}


class ConvMixer_1024_16Backbone(ConvMixerBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ConvMixerBackbone.from_preset("convmixer_1024_16", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}


class ConvMixer_512_16Backbone(ConvMixerBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ConvMixerBackbone.from_preset("convmixer_512_16", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include weights."""
        return {}


setattr(
    ConvMixer_1536_20Backbone,
    "__doc__",
    BASE_DOCSTRING.format(name="ConvMixer_1536_20"),
)
setattr(
    ConvMixer_1536_24Backbone,
    "__doc__",
    BASE_DOCSTRING.format(name="ConvMixer_1536_24"),
)
setattr(
    ConvMixer_768_32Backbone,
    "__doc__",
    BASE_DOCSTRING.format(name="ConvMixer_768_32"),
)
setattr(
    ConvMixer_1024_16Backbone,
    "__doc__",
    BASE_DOCSTRING.format(name="ConvMixer_1024_16"),
)
setattr(
    ConvMixer_512_16Backbone,
    "__doc__",
    BASE_DOCSTRING.format(name="ConvMixer_512_16"),
)
