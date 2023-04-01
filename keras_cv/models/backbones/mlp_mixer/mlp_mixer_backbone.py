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

"""MLP Mixer models for KerasCV.

Reference:
  - [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)
"""

import copy

from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers

from keras_cv.models import utils
from keras_cv.models.backbones.backbone import Backbone
from keras_cv.models.backbones.mlp_mixer.mlp_mixer_backbone_presets import (
    backbone_presets,
)
from keras_cv.utils.python_utils import classproperty

BASE_DOCSTRING = """Instantiates the {name} architecture.

    Reference:
        - [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)

    This class represents a Keras {name} model.

    For transfer learning use cases, make sure to read the [guide to transfer
        learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        include_rescaling: bool, whether or not to rescale the inputs. If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        include_top: bool, whether to include the fully-connected layer at the top of the
            network.  If provided, num_classes must be provided.
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
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
      A `keras.Model` instance.
"""


def apply_mlp_block(x, mlp_dim, name=None):
    """An MLP block consisting of two linear layers with GELU activation in
    between.

    Args:
      x: input tensor.
      mlp_dim: integer, the number of units to be present in the first layer.
      name: string, block label.

    Returns:
      the updated input tensor.
    """
    if name is None:
        name = f"mlp_block_{backend.get_uid('mlp_block')}"

    y = layers.Dense(mlp_dim, name=f"{name}_dense_1")(x)
    y = layers.Activation("gelu", name=f"{name}_gelu")(y)
    return layers.Dense(x.shape[-1], name=f"{name}_dense_2")(y)


def apply_mixer_block(x, tokens_mlp_dim, channels_mlp_dim, name=None):
    """A mixer block.

    Args:
      x: input tensor.
      tokens_mlp_dim: integer, number of units to be present in the MLP block
        dealing with tokens.
      channels_mlp_dim: integer, number of units to be present in the MLP block
        dealing with channels.
      name: string, block label.

    Returns:
      the updated input tensor.
    """
    if name is None:
        name = f"mixer_block_{backend.get_uid('mlp_block')}"

    y = layers.LayerNormalization()(x)
    y = layers.Permute((2, 1))(y)

    y = apply_mlp_block(y, tokens_mlp_dim, name=f"{name}_token_mixing")
    y = layers.Permute((2, 1))(y)
    x = layers.Add()([x, y])

    y = layers.LayerNormalization()(x)
    y = apply_mlp_block(y, channels_mlp_dim, name=f"{name}_channel_mixing")
    return layers.Add()([x, y])


@keras.utils.register_keras_serializable(package="keras_cv.models")
class MLPMixerBackbone(Backbone):

    """Instantiates the MLP Mixer architecture.

    Args:
      input_shape: tuple denoting the input shape, (224, 224, 3) for example.
      patch_size: integer denoting the size of the patches to be extracted
        from the inputs (16 for extracting 16x16 patches for example).
      num_blocks: integer, number of mixer blocks.
      hidden_dim: integer, dimension to which the patches will be linearly projected.
      tokens_mlp_dim: integer, dimension of the MLP block responsible for tokens.
      channels_mlp_dim: integer, dimension of the MLP block responsible for channels.
      include_rescaling: whether to rescale the inputs. If set to True,
        inputs will be passed through a `Rescaling(1/255.0)` layer.
      include_top: bool, whether to include the fully-connected
        layer at the top of the network. If provided, num_classes must be provided.
      num_classes: integer, optional number of classes to classify images
        into. Only to be specified if `include_top` is True.
      weights: one of `None` (random initialization) or a pretrained
        weight file path.
      input_tensor: optional Keras tensor (i.e., output of `layers.Input()`)
        to use as image input for the model.
      pooling: optional pooling mode for feature extraction
        when `include_top` is `False`.
        - `None` means that the output of the model will be
            the 4D tensor output of the
            last convolutional block.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional block, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.
      name: string, optional name to pass to the model, defaults to "MLPMixer".

    Returns:
      A `keras.Model` instance.
    """

    def __init__(
        self,
        *,
        input_shape,
        patch_size,
        num_blocks,
        hidden_dim,
        tokens_mlp_dim,
        channels_mlp_dim,
        include_rescaling,
        input_tensor=None,
        **kwargs,
    ):
        if not isinstance(input_shape, tuple):
            raise ValueError("`input_shape` needs to be tuple.")

        if len(input_shape) != 3:
            raise ValueError(
                "`input_shape` needs to contain dimensions for three"
                " axes: height, width, and channel ((224, 224, 3) for example)."
            )

        if input_shape[0] != input_shape[1]:
            raise ValueError("Non-uniform resolutions are not supported.")

        if input_shape[0] % patch_size != 0:
            raise ValueError(
                "Input resolution should be divisible by the patch size."
            )

        inputs = utils.parse_model_inputs(input_shape, input_tensor)

        x = inputs
        if include_rescaling:
            x = layers.Rescaling(1 / 255.0)(x)

        x = layers.Conv2D(
            filters=hidden_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="valid",
            name="patchify_and_projection",
        )(x)
        x = layers.Reshape((x.shape[1] * x.shape[2], x.shape[3]))(x)

        for i in range(num_blocks):
            x = apply_mixer_block(
                x, tokens_mlp_dim, channels_mlp_dim, name=f"mixer_block_{i}"
            )

        x = layers.LayerNormalization()(x)

        super().__init__(inputs=inputs, outputs=x, **kwargs)

        self.patch_size = patch_size
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.include_rescaling = include_rescaling
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape[1:],
                "patch_size": self.patch_size,
                "num_blocks": self.num_blocks,
                "hidden_dim": self.hidden_dim,
                "tokens_mlp_dim": self.tokens_mlp_dim,
                "channels_mlp_dim": self.channels_mlp_dim,
                "include_rescaling": self.include_rescaling,
                "input_tensor": self.input_tensor,
            }
        )
        return config

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy(backbone_presets)


class MLPMixerB16Backbone(MLPMixerBackbone):
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
        return MLPMixerBackbone.from_preset("mlpmixerb16", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}


class MLPMixerB32Backbone(MLPMixerBackbone):
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
        return MLPMixerBackbone.from_preset("mlpmixerb32", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}


class MLPMixerL16Backbone(MLPMixerBackbone):
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
        return MLPMixerBackbone.from_preset("mlpmixerl16", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}


setattr(
    MLPMixerB16Backbone, "__doc__", BASE_DOCSTRING.format(name="MLPMixerB16")
)
setattr(
    MLPMixerB32Backbone, "__doc__", BASE_DOCSTRING.format(name="MLPMixerB32")
)
setattr(
    MLPMixerL16Backbone, "__doc__", BASE_DOCSTRING.format(name="MLPMixerL16")
)
