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

import copy

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.layers.detectron2_layers import ViTDetPatchingAndEmbedding
from keras_cv.layers.detectron2_layers import WindowedTransformerEncoder
from keras_cv.layers.serializable_sequential import SerializableSequential
from keras_cv.models.backbones.backbone import Backbone
from keras_cv.models.backbones.detectron2.detectron2_backbone_presets import (
    backbone_presets,
)
from keras_cv.utils.python_utils import classproperty


@keras_cv_export("keras_cv.models.ViTDetBackbone")
class ViTDetBackbone(Backbone):
    """A ViT image encoder that uses a windowed transformer encoder and
    relative positional encodings.

    Args:
        img_size (int, optional): The size of the input image. Defaults to
            `1024`.
        patch_size (int, optional): the patch size to be supplied to the
            Patching layer to turn input images into a flattened sequence of
            patches. Defaults to `16`.
        in_chans (int, optional): The number of channels in the input image.
            Defaults to `3`.
        embed_dim (int, optional): The latent dimensionality to be projected
            into in the output of each stacked windowed transformer encoder.
            Defaults to `1280`.
        depth (int, optional): The number of transformer encoder layers to
            stack in the Vision Transformer. Defaults to `32`.
        mlp_dim (_type_, optional): The dimensionality of the hidden Dense
            layer in the transformer MLP head. Defaults to `1280*4`.
        num_heads (int, optional): the number of heads to use in the
            `MultiHeadAttentionWithRelativePE` layer of each transformer
            encoder. Defaults to `16`.
        out_chans (int, optional): The number of channels (features) in the
            output (image encodings). Defaults to `256`.
        use_bias (bool, optional): Whether to use bias to project the keys,
            queries, and values in the attention layer. Defaults to `True`.
        use_abs_pos (bool, optional): Whether to add absolute positional
            embeddings to the output patches. Defaults to `True`.
        use_rel_pos (bool, optional): Whether to use relative positional
            emcodings in the attention layer. Defaults to `False`.
        window_size (int, optional): The size of the window for windowed
            attention in the transformer encoder blocks. Defaults to `0`.
        global_attention_indices (list, optional): Indexes for blocks using
            global attention. Defaults to `[7, 15, 23, 31]`.
        layer_norm_epsilon (int, optional): The epsilon to use in the layer
            normalization blocks in transformer encoder. Defaults to `1e-6`.
    """

    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=1280,
        depth=32,
        mlp_dim=1280 * 4,
        num_heads=16,
        out_chans=256,
        use_bias=True,
        use_abs_pos=True,
        use_rel_pos=False,
        window_size=0,
        global_attention_indices=[7, 15, 23, 31],
        layer_norm_epsilon=1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.out_chans = out_chans
        self.use_bias = use_bias
        self.use_rel_pos = use_rel_pos
        self.use_abs_pos = use_abs_pos
        self.window_size = window_size
        self.global_attention_indices = global_attention_indices
        self.layer_norm_epsilon = layer_norm_epsilon

        self.patch_embed = ViTDetPatchingAndEmbedding(
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            embed_dim=embed_dim,
        )
        if self.use_abs_pos:
            self.pos_embed = self.add_weight(
                name="pos_embed",
                shape=(
                    1,
                    self.img_size // self.patch_size,
                    self.img_size // self.patch_size,
                    self.embed_dim,
                ),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.pos_embed = None
        self.transformer_blocks = []
        for i in range(depth):
            block = WindowedTransformerEncoder(
                project_dim=embed_dim,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                use_bias=use_bias,
                use_rel_pos=use_rel_pos,
                window_size=window_size
                if i not in global_attention_indices
                else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.transformer_blocks.append(block)
        self.bottleneck = SerializableSequential(
            [
                keras.layers.Conv2D(
                    filters=out_chans, kernel_size=1, use_bias=False
                ),
                keras.layers.LayerNormalization(epsilon=1e-6),
                keras.layers.Conv2D(
                    filters=out_chans,
                    kernel_size=3,
                    padding="same",
                    use_bias=False,
                ),
                keras.layers.LayerNormalization(epsilon=1e-6),
            ]
        )

        self.patch_embed.build(
            [None, self.img_size, self.img_size, self.in_chans]
        )
        self.bottleneck.build(
            [
                None,
                self.img_size // self.patch_size,
                self.img_size // self.patch_size,
                self.embed_dim,
            ]
        )

        self.built = True

    @property
    def pyramid_level_inputs(self):
        raise NotImplementedError(
            "The `ViTDetBackbone` model doesn't compute"
            " pyramid level features."
        )

    def call(self, x):
        B, _, _, _ = x.shape
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for block in self.transformer_blocks:
            x = block(x)
        return self.bottleneck(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "img_size": self.img_size,
                "patch_size": self.patch_size,
                "in_chans": self.in_chans,
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "mlp_dim": self.mlp_dim,
                "num_heads": self.num_heads,
                "out_chans": self.out_chans,
                "use_bias": self.use_bias,
                "use_abs_pos": self.use_abs_pos,
                "use_rel_pos": self.use_rel_pos,
                "window_size": self.window_size,
                "global_attention_indices": self.global_attention_indices,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy(backbone_presets)

    # @classproperty
    # def presets_with_weights(cls):
    #     """Dictionary of preset names and configurations that include
    #     weights."""
    #     return copy.deepcopy(backbone_presets)
