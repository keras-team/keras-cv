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

"""MiT backbone model.

References:

"""  # noqa: E501

from keras_cv import layers as cv_layers
from keras_cv.backend import keras
from keras_cv.models import utils
from keras_cv.models.backbones.backbone import Backbone
from keras_cv.models.backbones.mix_transformer.mix_transformer_backbone_presets import (  # noqa: E501
    backbone_presets,
)
from keras_cv.models.backbones.mix_transformer.mix_transformer_backbone_presets import (  # noqa: E501
    backbone_presets_with_weights,
)
from keras_cv.utils.python_utils import classproperty


@keras.saving.register_keras_serializable(package="keras_cv.models")
class MiT(Backbone):
    def __init__(
        self,
        input_shape=None,
        input_tensor=None,
        classes=None,
        include_top=None,
        embed_dims=None,
        depths=None,
        pooling=None,
        **kwargs,
    ):
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

        if include_top and as_backbone:
            raise ValueError(
                f"`as_backbone` must be `False` when `include_top=True`."
                f"Received as_backbone={as_backbone} and include_top={include_top}. "
            )

        drop_path_rate = 0.1
        dpr = [x.numpy() for x in tf.linspace(0.0, drop_path_rate, sum(depths))]
        blockwise_num_heads = [1, 2, 5, 8]
        blockwise_sr_ratios = [8, 4, 2, 1]
        num_stages = 4

        cur = 0
        patch_embedding_layers = []
        transformer_blocks = []
        layer_norms = []

        for i in range(num_stages):
            patch_embed_layer = OverlappingPatchingAndEmbedding(
                in_channels=input_shape[-1] if i == 0 else embed_dims[i - 1],
                out_channels=embed_dims[0] if i == 0 else embed_dims[i],
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                backend="tensorflow",
                name=f"patch_and_embed_{i}",
            )
            patch_embedding_layers.append(patch_embed_layer)

            transformer_block = [
                HierarchicalTransformerEncoder(
                    project_dim=embed_dims[i],
                    num_heads=blockwise_num_heads[i],
                    sr_ratio=blockwise_sr_ratios[i],
                    drop_prob=dpr[cur + k],
                    backend="tensorflow",
                    name=f"hierarchical_encoder_{i}_{k}",
                )
                for k in range(depths[i])
            ]
            transformer_blocks.append(transformer_block)
            cur += depths[i]
            layer_norms.append(keras.layers.LayerNormalization())

        inputs = utils.parse_model_inputs(input_shape, input_tensor)
        x = inputs

        batch_size = x.shape[0]
        pyramid_level_inputs = []
        for i in range(num_stages):
            x, H, W = patch_embedding_layers[i](x)
            for blk in transformer_blocks[i]:
                x = blk(x, H, W)
            x = layer_norms[i](x)
            C = x.shape[-1]
            x = x.reshape((batch_size, H, W, C))
            pyramid_level_inputs.append(x)

        super().__init__(
            inputs=inputs,
            outputs=x,
            **kwargs,
        )

        self.channels = embed_dims
        self.num_stages = num_stages
        self.output_channels = embed_dims
        self.classes = classes
        self.include_top = include_top
        self.pyramid_level_inputs = pyramid_level_inputs
        self.pooling = pooling

        self.patch_embedding_layers = []
        self.transformer_blocks = []

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "num_stages": self.num_stages,
                "output_channels": self.output_channels,
                "classes": self.classes,
                "include_top": self.include_top,
                "as_backbone": self.as_backbone,
                "pooling": self.pooling,
            }
        )
        return config
