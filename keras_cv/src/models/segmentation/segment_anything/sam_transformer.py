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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.models.segmentation.segment_anything.sam_layers import (
    MultiHeadAttentionWithDownsampling,
)
from keras_cv.src.models.segmentation.segment_anything.sam_layers import (
    TwoWayMultiHeadAttention,
)


@keras_cv_export("keras_cv.models.TwoWayTransformer", package="keras_cv.models")
class TwoWayTransformer(keras.layers.Layer):
    """A two-way cross-attention transformer decoder.

    A transformer decoder that attends to an input image using
    queries whose positional embedding is supplied.

    The transformer decoder design is shown in [1]_. Each decoder layer
    performs 4 steps: (1) self-attention on the tokens, (2) cross-attention
    from tokens (as queries) to the image embedding, (3) a point-wise MLP
    updates each token, and (4) cross-attention from the image embedding (as
    queries) to tokens. This last step updates the image embedding with prompt
    information. Each self/cross-attention and MLP has a residual connection
    and layer normalization.

    To ensure the decoder has access to critical geometric information the
    positional encodings are added to the image embedding whenever they
    participate in an attention layer. Additionally, the entire original
    prompt tokens (including their positional encodings) are re-added to the
    updated tokens whenever they participate in an attention layer. This
    allows for a strong dependence on both the prompt token's geometric
    location and type.

    Args:
        depth (int, optional): The depth of the attention blocks (the number
            of attention blocks to use). Defaults to `2`.
        embed_dim (int, optional): The number of features of the input image
            and point embeddings. Defaults to `256`.
        num_heads (int, optional): Number of heads to use in the attention
            layers. Defaults to `8`.
        mlp_dim (int, optional): The number of units in the hidden layer of
            the MLP block used in the attention layers. Defaults to `2048`.
        activation (str, optional): The activation of the MLP block's output
            layer used in the attention layers. Defaults to `"relu"`.
        attention_downsample_rate (int, optional): The downsample rate of the
            attention layers. Defaults to `2`.

    References:
        - [Segment Anything paper](https://arxiv.org/abs/2304.02643)
        - [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything)
    """  # noqa: E501

    def __init__(
        self,
        *,
        depth=2,
        embed_dim=256,
        num_heads=8,
        mlp_dim=2048,
        activation="relu",
        attention_downsample_rate=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.activation = activation
        self.attention_downsample_rate = attention_downsample_rate
        self.layers = []
        for i in range(depth):
            self.layers.append(
                TwoWayMultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=embed_dim // num_heads,
                    mlp_dim=mlp_dim,
                    skip_first_layer_pe=(i == 0),
                    attention_downsample_rate=attention_downsample_rate,
                    activation=activation,
                )
            )
        self.final_attention_token_to_image = (
            MultiHeadAttentionWithDownsampling(
                num_heads=num_heads,
                key_dim=embed_dim // num_heads,
                downsample_rate=attention_downsample_rate,
            )
        )
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5)

    def build(self, input_shape=None):
        for layer in self.layers:
            layer.build()
        self.final_attention_token_to_image.build()
        self.final_layer_norm.build([None, None, self.embed_dim])
        self.built = True

    def call(self, image_embedding, image_pe, point_embedding):
        shape = ops.shape(image_embedding)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]
        image_embedding = ops.reshape(image_embedding, (B, H * W, C))

        shape = ops.shape(image_pe)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]
        image_pe = ops.reshape(image_pe, (B, H * W, C))
        queries = point_embedding
        keys = image_embedding

        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        queries_with_pe = queries + point_embedding
        keys_with_pe = keys + image_pe
        attention_map = self.final_attention_token_to_image(
            query=queries_with_pe, key=keys_with_pe, value=keys
        )
        queries = queries + attention_map
        queries = self.final_layer_norm(queries)

        return queries, keys

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depth": self.depth,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "activation": self.activation,
                "attention_downsample_rate": self.attention_downsample_rate,
            }
        )
        return config
