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

import copy

import numpy as np

from keras_cv import layers as cv_layers
from keras_cv.backend import keras
from keras_cv.backend import ops
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
class MiTBackbone(Backbone):
    def __init__(
        self,
        include_rescaling,
        input_shape=(None, None, 3),
        input_tensor=None,
        embed_dims=None,
        depths=None,
        **kwargs,
    ):
        drop_path_rate = 0.1
        dpr = [x for x in np.linspace(0.0, drop_path_rate, sum(depths))]
        blockwise_num_heads = [1, 2, 5, 8]
        blockwise_sr_ratios = [8, 4, 2, 1]
        num_stages = 4

        cur = 0
        patch_embedding_layers = []
        transformer_blocks = []
        layer_norms = []

        for i in range(num_stages):
            patch_embed_layer = cv_layers.OverlappingPatchingAndEmbedding(
                out_channels=embed_dims[0] if i == 0 else embed_dims[i],
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                name=f"patch_and_embed_{i}",
            )
            patch_embedding_layers.append(patch_embed_layer)

            transformer_block = [
                cv_layers.HierarchicalTransformerEncoder(
                    project_dim=embed_dims[i],
                    num_heads=blockwise_num_heads[i],
                    sr_ratio=blockwise_sr_ratios[i],
                    drop_prob=dpr[cur + k],
                    name=f"hierarchical_encoder_{i}_{k}",
                )
                for k in range(depths[i])
            ]
            transformer_blocks.append(transformer_block)
            cur += depths[i]
            layer_norms.append(keras.layers.LayerNormalization())

        inputs = utils.parse_model_inputs(input_shape, input_tensor)
        x = inputs

        if include_rescaling:
            x = keras.layers.Rescaling(scale=1 / 255)(x)

        pyramid_level_inputs = []
        for i in range(num_stages):
            # Compute new height/width after the `proj`
            # call in `OverlappingPatchingAndEmbedding`
            stride = 4 if i == 0 else 2
            new_height, new_width = (
                int(ops.shape(x)[1] / stride),
                int(ops.shape(x)[2] / stride),
            )

            x = patch_embedding_layers[i](x)
            for blk in transformer_blocks[i]:
                x = blk(x)
            x = layer_norms[i](x)
            C = x.shape[-1]
            x = CustomReshaping(new_height, new_width)(x)
            pyramid_level_inputs.append(x)

        super().__init__(inputs=inputs, outputs=x, **kwargs)

        self.num_stages = num_stages
        self.output_channels = embed_dims
        self.pyramid_level_inputs = {
            f"P{i + 1}": name for i, name in enumerate(pyramid_level_inputs)
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "num_stages": self.num_stages,
                "output_channels": self.output_channels,
                "pyramid_level_inputs": self.pyramid_level_inputs,
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


@keras.saving.register_keras_serializable(package="keras_cv")
class CustomReshaping(keras.layers.Layer):
    def __init__(self, H, W):
        super().__init__()
        self.H = H
        self.W = W

    def call(self, x):
        input_shape = ops.shape(x)
        x = ops.reshape(x, (input_shape[0], self.H, self.W, input_shape[-1]))
        return x
