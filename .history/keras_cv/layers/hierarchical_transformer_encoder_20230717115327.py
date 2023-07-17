# Copyright 2023 David Landup
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

import tensorflow as tf
from torch import nn

from deepvision.layers.efficient_attention import EfficientMultiheadAttention
from deepvision.layers.mix_ffn import MixFFN
from deepvision.layers.stochasticdepth import StochasticDepth


class HierarchicalTransformerEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        project_dim,
        num_heads,
        sr_ratio=1,
        drop_prob=0.0,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm1 = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon
        )
        self.attn = EfficientMultiheadAttention(
            project_dim, num_heads, sr_ratio, backend="tensorflow"
        )
        self.drop_path = StochasticDepth(drop_prob, backend="tensorflow")
        self.norm2 = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon
        )
        self.mlp = MixFFN(
            channels=project_dim,
            mid_channels=int(project_dim * 4),
            backend="tensorflow",
        )

    def call(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x
