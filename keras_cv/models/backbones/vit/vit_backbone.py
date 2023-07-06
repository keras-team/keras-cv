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
"""ViT (Vision Transformer) models for Keras.
Reference:
  - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929v2)
    (ICLR 2021)
  - [How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers](https://arxiv.org/abs/2106.10270)
    (CoRR 2021)
"""  # noqa: E501

import copy

from tensorflow import keras
from tensorflow.keras import layers

from keras_cv.layers import TransformerEncoder
from keras_cv.layers.vit_layers import PatchingAndEmbedding
from keras_cv.models import utils
from keras_cv.models.backbones.backbone import Backbone
from keras_cv.models.backbones.vit.vit_backbone_presets import backbone_presets
from keras_cv.models.backbones.vit.vit_backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.utils.python_utils import classproperty


@keras.utils.register_keras_serializable(package="keras_cv.models")
class ViTBackbone(Backbone):
    """Instantiates the ViT architecture.

    Reference:
        - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929v2)
        (ICLR 2021)
    This function returns a Keras {name} model.

    The naming convention of ViT models follows: ViTSize_Patch-size
        (i.e. ViTS16).
    The following sizes were released in the original paper:
        - S (Small)
        - B (Base)
        - L (Large)
    But subsequent work from the same authors introduced:
        - Ti (Tiny)
        - H (Huge)

    The parameter configurations for all of these sizes, at patch sizes 16 and
    32 are made available, following the naming convention laid out above.

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        mlp_dim: the dimensionality of the hidden Dense layer in the transformer
            MLP head
        include_rescaling: bool, whether to rescale the inputs. If set to
            True, inputs will be passed through a `Rescaling(1/255.0)` layer.
        name: string, model name.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        project_dim: the latent dimensionality to be projected into in the
            output of each stacked transformer encoder
        activation: the activation function to use in the first `layers.Dense`
            layer in the MLP head of the transformer encoder
        attention_dropout: the dropout rate to apply to the `MultiHeadAttention`
            in each transformer encoder
        mlp_dropout: the dropout rate to apply between `layers.Dense` layers
            in the MLP head of the transformer encoder
        num_heads: the number of heads to use in the `MultiHeadAttention` layer
            of each transformer encoder
        transformer_layer_num: the number of transformer encoder layers to stack
            in the Vision Transformer
        patch_size: the patch size to be supplied to the Patching layer to turn
            input images into a flattened sequence of patches
        **kwargs: Pass-through keyword arguments to `keras.Model`.
    """  # noqa: E501

    def __init__(
        self,
        *,
        include_rescaling=False,
        input_shape=(224, 224, 3),
        input_tensor=None,
        patch_size=None,
        transformer_layer_num=None,
        num_heads=None,
        mlp_dropout=None,
        attention_dropout=None,
        activation=None,
        project_dim=None,
        mlp_dim=None,
        **kwargs,
    ):

        inputs = utils.parse_model_inputs(input_shape, input_tensor)
        x = inputs

        if include_rescaling:
            x = layers.Rescaling(1.0 / 255.0, name="rescaling")(x)

        # The previous layer rescales [0..255] to [0..1] if applicable
        # This one rescales [0..1] to [-1..1] since ViTs expect [-1..1]
        x = layers.Rescaling(scale=1.0 / 0.5, offset=-1.0, name="rescaling_2")(
            x
        )

        encoded_patches = PatchingAndEmbedding(project_dim, patch_size)(x)
        encoded_patches = layers.Dropout(mlp_dropout)(encoded_patches)

        for encoder_num in range(transformer_layer_num):
            encoded_patches = TransformerEncoder(
                project_dim=project_dim,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                mlp_dropout=mlp_dropout,
                attention_dropout=attention_dropout,
                activation=activation or keras.activations.gelu,
            )(encoded_patches)
        output = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create model.
        super().__init__(inputs=inputs, outputs=output, **kwargs)

        self.include_rescaling = include_rescaling
        self.input_tensor = input_tensor
        self.patch_size = patch_size
        self.transformer_layer_num = transformer_layer_num
        self.num_heads = num_heads
        self.mlp_dropout = mlp_dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.project_dim = project_dim
        self.mlp_dim = mlp_dim

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "include_rescaling": self.include_rescaling,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "patch_size": self.patch_size,
                "transformer_layer_num": self.transformer_layer_num,
                "num_heads": self.num_heads,
                "mlp_dropout": self.mlp_dropout,
                "attention_dropout": self.attention_dropout,
                "activation": self.activation,
                "project_dim": self.project_dim,
                "mlp_dim": self.mlp_dim,
                "trainable": self.trainable,
            }
        )
        return config

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy(backbone_presets)

    @classproperty
    def presets_with_weights(cls):
        """
        Dictionary of preset names and configurations
        that include weights.
        """
        return copy.deepcopy(backbone_presets_with_weights)
