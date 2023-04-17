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

"""ConvMixer backbone model for Keras.

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

    Reference:
        - [Patches Are All You Need?](https://arxiv.org/abs/2201.09792)

    For transfer learning use cases, make sure to read the [guide to transfer
        learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        dim: integer, number of filters.
        depth: integer, number of ConvMixer Layer.
        patch_size: integer, size of the patches.
        kernel_size: integer, kernel size for Conv2D layers.
        include_rescaling: bool, whether to rescale the inputs. If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e., output of `layers.Input()`)
            to use as image input for the model.

    Examples:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Pretrained backbone
    model = keras_cv.models.ConvMixerBackbone.from_preset(
        "convmixer_512_16_imagenet"
    )
    output = model(input_data)

    # Randomly initialized backbone with a custom config
    model = ConvMixerBackbone(
        dim=512,
        depth=16,
        patch_size=7,
        kernel_size=8,
        include_rescaling=False,
    )
    output = model(input_data)
    ```
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

        pyramid_level_inputs = {}
        for block_index in range(depth):
            x = apply_conv_mixer_layer(x, dim, kernel_size)
            pyramid_level_inputs[block_index] = x.node.layer.name

        super().__init__(inputs=inputs, outputs=x, **kwargs)

        self.pyramid_level_inputs = pyramid_level_inputs
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
        """Dictionary of preset names and configurations that include
        weights."""
        return copy.deepcopy(backbone_presets_with_weights)


ALIAS_DOCSTRING = """ConvMixerBackbone model with {num_layers} layers with
    {channels} output channels.

    Reference:
        - [Patches Are All You Need?](https://arxiv.org/abs/2201.09792)

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        include_rescaling: bool, whether to rescale the inputs. If set to
            True, inputs will be passed through a `Rescaling(1/255.0)` layer.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e., output of `layers.Input()`)

    Examples:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Randomly initialized backbone
    model = ConvMixer_{channels}_{num_layers}Backbone()
    output = model(input_data)
    ```
"""  # noqa: E501


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
        return {
            "convmixer_512_16_imagenet": copy.deepcopy(
                backbone_presets["convmixer_512_16_imagenet"]
            )
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


setattr(
    ConvMixer_1536_20Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(num_layers="20", channels="1536"),
)
setattr(
    ConvMixer_1536_24Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(num_layers="24", channels="1536"),
)
setattr(
    ConvMixer_768_32Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(num_layers="32", channels="768"),
)
setattr(
    ConvMixer_1024_16Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(num_layers="16", channels="1024"),
)
setattr(
    ConvMixer_512_16Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(num_layers="16", channels="512"),
)
