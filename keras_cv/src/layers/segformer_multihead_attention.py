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


@keras_cv_export("keras_cv.layers.SegFormerMultiheadAttention")
class SegFormerMultiheadAttention(keras.layers.Layer):
    def __init__(self, project_dim, num_heads, sr_ratio):
        """
        Efficient MultiHeadAttention implementation as a Keras layer.
        A huge bottleneck in scaling transformers is the self-attention layer
        with an O(n^2) complexity.

        SegFormerMultiheadAttention performs a sequence reduction (SR) operation
        with a given ratio, to reduce the sequence length before performing key and value projections,
        reducing the O(n^2) complexity to O(n^2/R) where R is the sequence reduction ratio.

        References:
        - [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) (CVPR 2021) # noqa: E501
        - [NVlabs' official implementation](https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/backbones/mix_transformer.py) # noqa: E501
        - [@sithu31296's reimplementation](https://github.com/sithu31296/semantic-segmentation/blob/main/semseg/models/backbones/mit.py) # noqa: E501
        - [Ported from the TensorFlow implementation from DeepVision](https://github.com/DavidLandup0/deepvision/blob/main/deepvision/layers/efficient_attention.py) # noqa: E501

        Args:
            project_dim: integer, the dimensionality of the projection
                of the `SegFormerMultiheadAttention` layer.
            num_heads: integer, the number of heads to use in the
                attention computation.
            sr_ratio: integer, the sequence reduction ratio to perform
                on the sequence before key and value projections.

        Example:

        ```
        tensor = tf.random.uniform([1, 196, 32])
        output = keras_cv.layers.SegFormerMultiheadAttention(project_dim=768,
                                                            num_heads=2,
                                                            sr_ratio=4)(tensor)
        print(output.shape) # (1, 196, 32)
        ```
        """
        super().__init__()
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.scale = (project_dim // num_heads) ** -0.5
        self.q = keras.layers.Dense(project_dim)
        self.k = keras.layers.Dense(project_dim)
        self.v = keras.layers.Dense(project_dim)
        self.proj = keras.layers.Dense(project_dim)

        if sr_ratio > 1:
            self.sr = keras.layers.Conv2D(
                filters=project_dim,
                kernel_size=sr_ratio,
                strides=sr_ratio,
                padding="same",
            )
            self.norm = keras.layers.LayerNormalization()

    def call(self, x):
        input_shape = ops.shape(x)
        H, W = int(math.sqrt(input_shape[1])), int(math.sqrt(input_shape[1]))
        B, C = input_shape[0], input_shape[2]

        q = self.q(x)
        q = ops.reshape(
            q,
            (
                input_shape[0],
                input_shape[1],
                self.num_heads,
                input_shape[2] // self.num_heads,
            ),
        )
        q = ops.transpose(q, [0, 2, 1, 3])

        if self.sr_ratio > 1:
            x = ops.reshape(
                ops.transpose(x, [0, 2, 1]),
                (B, H, W, C),
            )
            x = self.sr(x)
            x = ops.reshape(x, [input_shape[0], input_shape[2], -1])
            x = ops.transpose(x, [0, 2, 1])
            x = self.norm(x)

        k = self.k(x)
        v = self.v(x)

        k = ops.transpose(
            ops.reshape(
                k,
                [B, -1, self.num_heads, C // self.num_heads],
            ),
            [0, 2, 1, 3],
        )

        v = ops.transpose(
            ops.reshape(
                v,
                [B, -1, self.num_heads, C // self.num_heads],
            ),
            [0, 2, 1, 3],
        )

        attn = (q @ ops.transpose(k, [0, 1, 3, 2])) * self.scale
        attn = ops.nn.softmax(attn, axis=-1)

        attn = attn @ v
        attn = ops.reshape(
            ops.transpose(attn, [0, 2, 1, 3]),
            [input_shape[0], input_shape[1], input_shape[2]],
        )

        x = self.proj(attn)
        return x
