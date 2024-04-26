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

import math

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.layers.regularization.drop_path import DropPath
from keras_cv.src.layers.segformer_multihead_attention import (
    SegFormerMultiheadAttention,
)


@keras_cv_export("keras_cv.layers.HierarchicalTransformerEncoder")
class HierarchicalTransformerEncoder(keras.layers.Layer):
    """
    Hierarchical transformer encoder block implementation as a Keras Layer.
    The layer uses `SegFormerMultiheadAttention` as a `MultiHeadAttention`
    alternative for computational efficiency, and is meant to be used
    within the SegFormer architecture.

    References:
        - [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) (CVPR 2021) # noqa: E501
        - [Official PyTorch implementation](https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/backbones/mix_transformer.py) # noqa: E501
        - [Ported from the TensorFlow implementation from DeepVision](https://github.com/DavidLandup0/deepvision/blob/main/deepvision/layers/hierarchical_transformer_encoder.py) # noqa: E501

    Args:
        project_dim: integer, the dimensionality of the projection of the
            encoder, and output of the `SegFormerMultiheadAttention` layer.
            Due to the residual addition the input dimensionality has to be
            equal to the output dimensionality.
        num_heads: integer, the number of heads for the
            `SegFormerMultiheadAttention` layer.
        drop_prob: float, the probability of dropping a random
            sample using the `DropPath` layer. Defaults to `0.0`.
        layer_norm_epsilon: float, the epsilon for
            `LayerNormalization` layers. Defaults to `1e-06`
        sr_ratio: integer, the ratio to use within
            `SegFormerMultiheadAttention`. If set to > 1, a `Conv2D`
             layer is used to reduce the length of the sequence. Defaults to `1`.

    Example:

    ```
    project_dim = 1024
    num_heads = 4
    patch_size = 16

    encoded_patches = keras_cv.layers.OverlappingPatchingAndEmbedding(
    project_dim=project_dim, patch_size=patch_size)(img_batch)

    trans_encoded = keras_cv.layers.HierarchicalTransformerEncoder(project_dim=project_dim,
                                                                   num_heads=num_heads,
                                                                   sr_ratio=1)(encoded_patches)

    print(trans_encoded.shape) # (1, 3136, 1024)
    ```
    """

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
        self.project_dim = project_dim
        self.num_heads = num_heads
        self.drop_prop = drop_prob

        self.norm1 = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)
        self.attn = SegFormerMultiheadAttention(
            project_dim, num_heads, sr_ratio
        )
        self.drop_path = DropPath(drop_prob)
        self.norm2 = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)
        self.mlp = self.MixFFN(
            channels=project_dim,
            mid_channels=int(project_dim * 4),
        )

    def build(self, input_shape):
        super().build(input_shape)
        self.H = ops.sqrt(ops.cast(input_shape[1], "float32"))
        self.W = ops.sqrt(ops.cast(input_shape[2], "float32"))

    def call(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mlp": keras.saving.serialize_keras_object(self.mlp),
                "project_dim": self.project_dim,
                "num_heads": self.num_heads,
                "drop_prop": self.drop_prop,
            }
        )
        return config

    class MixFFN(keras.layers.Layer):
        def __init__(self, channels, mid_channels):
            super().__init__()
            self.fc1 = keras.layers.Dense(mid_channels)
            self.dwconv = keras.layers.DepthwiseConv2D(
                kernel_size=3,
                strides=1,
                padding="same",
            )
            self.fc2 = keras.layers.Dense(channels)

        def call(self, x):
            x = self.fc1(x)
            shape = ops.shape(x)
            H, W = int(math.sqrt(shape[1])), int(math.sqrt(shape[1]))
            B, C = shape[0], shape[2]
            x = ops.reshape(x, (B, H, W, C))
            x = self.dwconv(x)
            x = ops.reshape(x, (B, -1, C))
            x = ops.nn.gelu(x)
            x = self.fc2(x)
            return x
