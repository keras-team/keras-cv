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


class __HierarchicalTransformerEncoderPT(nn.Module):
    def __init__(
        self,
        project_dim,
        num_heads,
        sr_ratio=1,
        drop_prob=0.0,
        layer_norm_epsilon=1e-6,
        name=None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(project_dim)
        self.attn = EfficientMultiheadAttention(
            project_dim, num_heads, sr_ratio
        )
        self.drop_path = StochasticDepth(drop_prob, backend="pytorch")
        self.norm2 = nn.LayerNorm(project_dim, eps=layer_norm_epsilon)
        self.mlp = MixFFN(
            channels=project_dim,
            mid_channels=int(project_dim * 4),
            backend="pytorch",
        )

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class __HierarchicalTransformerEncoderTF(tf.keras.layers.Layer):
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


LAYER_BACKBONES = {
    "tensorflow": __HierarchicalTransformerEncoderTF,
    "pytorch": __HierarchicalTransformerEncoderPT,
}


def HierarchicalTransformerEncoder(
    project_dim,
    num_heads,
    sr_ratio,
    drop_prob=0.0,
    layer_norm_epsilon=1e-6,
    backend=None,
    name=None,
):
    """
    TransformerEncoder variant, which uses `deepvision.layers.EfficientMultiheadAttention` in lieu of `torch.nn.MultiheadAttention` or `tf.keras.layers.MultiHeadAttention`.
    `EfficientMultiheadAttention` shorten the sequence they operate on by a reduction factor, to reduce computational cost.
    The `HierarchicalTransformerEncoder` is designed to encode feature maps at multiple spatial levels, similar to how CNNs encode multiple spatial levels.

    Reference:
        - ["SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"](https://arxiv.org/pdf/2105.15203v2.pdf)

    Args:
        project_dim: the dimensionality of the projection of the encoder, and output of the `EfficientMultiheadAttention`
        num_heads: the number of heads for the `EfficientMultiheadAttention` layer
        sr_ratio: the reduction ratio to apply within the `EfficientMultiheadAttention` layer
        layer_norm_epsilon: default 1e-06, the epsilon for Layer Normalization layers
        drop_prob: the drop probability for the `DropPath` layers
        backend: the backend framework to use

    Basic usage:

    ```
    # (B, SEQ_LEN, CHANNELS)
    inp = torch.rand(1, 3136, 32)
    H, W = 56

    output = deepvision.layers.HierarchicalTransformerEncoder(project_dim=32,
                                                              num_heads=2,
                                                              sr_ratio=4,
                                                              backend='pytorch')(inp, H, W)
    print(output.shape) # torch.Size([1, 3136, 32])
    ```
    """

    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    layer = layer_class(
        project_dim=project_dim,
        num_heads=num_heads,
        sr_ratio=sr_ratio,
        drop_prob=drop_prob,
        layer_norm_epsilon=layer_norm_epsilon,
        name=name,
    )

    return layer
