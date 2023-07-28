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

from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.models.segmentation.segment_anything.sam_layers import MLPBlock


@keras.utils.register_keras_serializable(package="keras_cv")
class MultiHeadAttentionWithDownsampling(keras.layers.Layer):
    """Multi-Head Attention with downsampling.

    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.

    This layer first downscales the features of input queries, keys, and
    values using a dense layer. Multi-head attention is then performed
    and the attention map is projected back (upscaled) to the number of
    input features.

    Args:
        num_heads (int): Number of attention heads.
        key_dim (int): Size of each attention head for query, key, and
            value.
        downsample_rate (int, optional): The factor by which to downscale the
            input features i.e. the input features of size `key_dim` are
            projected down to `key_dim // downsample_rate`.
    """

    def __init__(self, num_heads, key_dim, downsample_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.downsample_rate = downsample_rate
        self.internal_dims = key_dim // downsample_rate

        # Downsample
        self.query_proj = keras.layers.Dense(
            self.internal_dims * self.num_heads
        )
        self.key_proj = keras.layers.Dense(self.internal_dims * self.num_heads)
        self.value_proj = keras.layers.Dense(
            self.internal_dims * self.num_heads
        )

        # Upsample
        self.out_proj = keras.layers.Dense(self.key_dim * self.num_heads)

        self.built = False

    def __separate_heads(self, x):
        B, N, C = x.shape
        x = ops.reshape(x, (B, N, self.num_heads, C // self.num_heads))
        return ops.transpose(x, axes=(0, 2, 1, 3))

    def __recombine_heads(self, x):
        B, N_H, N_T, C_PH = x.shape
        x = ops.transpose(x, axes=(0, 2, 1, 3))
        return ops.reshape(x, (B, N_T, N_H * C_PH))

    def build(self, query_shape, value_shape, key_shape):
        self.query_proj.build(query_shape)
        self.key_proj.build(key_shape)
        self.value_proj.build(value_shape)
        self.out_proj.build([self.internal_dims * self.num_heads])

        self.built = True

    def call(self, query, value, key):
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        # Separate into heads
        query = self.__separate_heads(query)
        key = self.__separate_heads(key)
        value = self.__separate_heads(value)

        # Attention
        C_PH = query.shape[-1]
        out = query @ ops.transpose(key, (0, 1, 3, 2))
        out = out / ops.sqrt(ops.cast(C_PH, dtype=self.dtype))
        out = ops.softmax(out, axis=-1)

        # Get output
        attention_map = out @ value
        attention_map = self.__recombine_heads(attention_map)
        return self.out_proj(attention_map)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "downsample_rate": self.downsample_rate,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="keras_cv")
class TwoWayMultiHeadAttention(keras.layers.Layer):
    """Two-way multi-head attention layer.

    Args:
        num_heads (int): Number of attention heads.
        key_dim (int): Size of each attention head for query, key, and
            value.
        mlp_dim (int): Number of hidden dims to use in the mlp block.
        skip_first_layer_pe (bool): A boolean indicating whether to skip the
            first layer positional embeddings.
        attention_downsample_rate (int, optional): The downsample rate to use
            in the attention layers. Defaults to 2.
        activation (str, optional): The activation for the mlp block's output
            layer. Defaults to "relu".
    """

    def __init__(
        self,
        num_heads,
        key_dim,
        mlp_dim,
        skip_first_layer_pe,
        attention_downsample_rate=2,
        activation="relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mlp_dim = mlp_dim
        self.skip_first_layer_pe = skip_first_layer_pe
        self.attention_downsample_rate = attention_downsample_rate
        self.activation = activation

        self.self_attention = MultiHeadAttentionWithDownsampling(
            num_heads=num_heads, key_dim=key_dim
        )
        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.cross_attention_token_to_image = (
            MultiHeadAttentionWithDownsampling(
                num_heads=num_heads,
                key_dim=key_dim,
                downsample_rate=attention_downsample_rate,
            )
        )
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-5)

        self.mlp_block = MLPBlock(key_dim * num_heads, mlp_dim, activation)

        self.layer_norm3 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.cross_attention_image_to_token = (
            MultiHeadAttentionWithDownsampling(
                num_heads=num_heads,
                key_dim=key_dim,
                downsample_rate=attention_downsample_rate,
            )
        )
        self.layer_norm4 = keras.layers.LayerNormalization(epsilon=1e-5)

        self.built = False

    def build(self, queries_shape, keys_shape, query_pe_shape, key_pe_shape):
        self.self_attention.build(
            query_shape=queries_shape,
            value_shape=queries_shape,
            key_shape=queries_shape,
        )
        self.layer_norm1.build(queries_shape)
        self.cross_attention_token_to_image.build(
            query_shape=queries_shape,
            key_shape=keys_shape,
            value_shape=keys_shape,
        )
        self.layer_norm2.build(queries_shape)
        self.mlp_block.build(queries_shape)
        self.layer_norm3.build(queries_shape)
        self.cross_attention_image_to_token.build(
            query_shape=keys_shape,
            key_shape=queries_shape,
            value_shape=queries_shape,
        )
        self.layer_norm4.build(keys_shape)

        self.built = True

    def call(self, queries, keys, query_pe, key_pe):
        # print("Actual queries_shape:", queries.shape)
        if self.skip_first_layer_pe:
            queries = self.self_attention(
                query=queries, value=queries, key=queries
            )
        else:
            queries_with_pe = queries + query_pe
            attention_map = self.self_attention(
                query=queries_with_pe, key=queries_with_pe, value=queries
            )
            queries = queries + attention_map
        queries = self.layer_norm1(queries)

        queries_with_pe = queries + query_pe
        keys_with_pe = keys + key_pe
        attention_map = self.cross_attention_token_to_image(
            query=queries_with_pe, key=keys_with_pe, value=keys
        )
        queries = queries + attention_map
        queries = self.layer_norm2(queries)

        mlp_out = self.mlp_block(queries)
        queries = queries + mlp_out
        queries = self.layer_norm3(queries)

        queries_with_pe = queries + query_pe
        keys_with_pe = keys + key_pe
        attention_map = self.cross_attention_image_to_token(
            query=keys_with_pe, key=queries_with_pe, value=queries
        )
        keys = keys + attention_map
        keys = self.layer_norm4(keys)

        return queries, keys

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "mlp_dim": self.mlp_dim,
                "skip_first_layer_pe": self.skip_first_layer_pe,
                "attention_downsample_rate": self.attention_downsample_rate,
                "activation": self.activation,
            }
        )
        return config


@keras.utils.register_keras_serializable(package="keras_cv")
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
        depth (int): The depth of the attention blocks (the number
            of attention blocks to use).
        embedding_dim (int): The number of features of the input image and
            point embeddings.
        num_heads (int): Number of heads to use in the attention layers.
        mlp_dim (int): The number of units in the hidden layer of the MLP
            block used in the attention layers.
        activation (str, optional): The activation of the MLP block's output
            layer used in the attention layers. Defaults to "relu".
        attention_downsample_rate (int, optional): The downsample rate of the
            attention layers. Defaults to 2.

    References:
        - [Segment Anything](https://arxiv.org/abs/2304.02643)
    """

    def __init__(
        self,
        depth,
        embedding_dim,
        num_heads,
        mlp_dim,
        activation="relu",
        attention_downsample_rate=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.activation = activation
        self.attention_downsample_rate = attention_downsample_rate
        self.layers = []
        for i in range(depth):
            self.layers.append(
                TwoWayMultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=embedding_dim // num_heads,
                    mlp_dim=mlp_dim,
                    skip_first_layer_pe=(i == 0),
                    attention_downsample_rate=attention_downsample_rate,
                    activation=activation,
                )
            )
        self.final_attention_token_to_image = (
            MultiHeadAttentionWithDownsampling(
                num_heads=num_heads,
                key_dim=embedding_dim // num_heads,
                downsample_rate=attention_downsample_rate,
            )
        )
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5)

        self.built = False

    def build(
        self, image_embedding_shape, image_pe_shape, point_embedding_shape
    ):
        B, H, W, C = image_embedding_shape
        image_embedding_shape = [B, H * W, C]
        for layer in self.layers:
            layer.build(
                queries_shape=point_embedding_shape,
                keys_shape=image_embedding_shape,
                query_pe_shape=point_embedding_shape,
                key_pe_shape=image_embedding_shape,
            )
        self.final_attention_token_to_image.build(
            query_shape=point_embedding_shape,
            key_shape=image_embedding_shape,
            value_shape=image_embedding_shape,
        )
        self.final_layer_norm.build(point_embedding_shape)

        self.built = True

    def call(self, image_embedding, image_pe, point_embedding):
        B, H, W, C = image_embedding.shape
        image_embedding = ops.reshape(image_embedding, (B, H * W, C))
        B, H, W, C = image_pe.shape
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
                "embedding_dim": self.embedding_dim,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "activation": self.activation,
                "attention_downsample_rate": self.attention_downsample_rate,
            }
        )
        return config
