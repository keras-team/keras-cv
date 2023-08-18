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

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.models.segmentation.segment_anything.sam_layers import MLPBlock


@keras_cv_export("keras_cv.layers.MultiHeadAttentionWithRelativePE")
class MultiHeadAttentionWithRelativePE(keras.layers.Layer):
    """Multi-head Attention block with relative position embeddings.

    Args:
        num_heads (int): Number of attention heads.
        key_dim (int): Size of each attention head for query, key, and
            value.
        use_bias (bool, optional): Whether to use bias when projecting
            the queries, keys, and values. Defaults to `True`.
        use_rel_pos (bool, optional): Whether to use relative positional
            embeddings or not. Defaults to `False`.
        input_size (tuple[int, int], optional): Size of the input image.
            Must be provided when using relative positional embeddings.
            Defaults to `None`.

    Raises:
        ValueError: When `input_size = None` with `use_rel_pos = True`.
    """

    def __init__(
        self,
        num_heads,
        key_dim,
        use_bias=True,
        use_rel_pos=False,
        input_size=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.scale = self.key_dim**-0.5
        self.use_bias = use_bias

        self.qkv = keras.layers.Dense(
            key_dim * self.num_heads * 3, use_bias=self.use_bias
        )
        self.projection = keras.layers.Dense(key_dim * self.num_heads)

        self.input_size = input_size
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            if input_size is None:
                raise ValueError(
                    "Input size must be provided if using relative "
                    "positional encoding."
                )
            self.rel_pos_h = self.add_weight(
                name="rel_pos_h",
                shape=(2 * self.input_size[0] - 1, self.key_dim),
                initializer="zeros",
                trainable=True,
            )
            self.rel_pos_w = self.add_weight(
                name="rel_pos_w",
                shape=(2 * self.input_size[1] - 1, self.key_dim),
                initializer="zeros",
                trainable=True,
            )

        self.qkv.build([self.key_dim * self.num_heads])
        self.projection.build([self.key_dim * self.num_heads])

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        B, H, W, C = x.shape
        qkv = ops.transpose(
            ops.reshape(
                self.qkv(x), (B, H * W, 3, self.num_heads, self.key_dim)
            ),
            axes=(2, 0, 3, 1, 4),
        )
        qkv = ops.reshape(qkv, (3, B * self.num_heads, H * W, self.key_dim))
        queries, keys, values = ops.unstack(qkv, axis=0)
        attention_map = (queries * self.scale) @ ops.transpose(
            keys, axes=(0, 2, 1)
        )

        if self.use_rel_pos:
            attention_map = add_decomposed_rel_pos(
                attention_map,
                queries,
                self.rel_pos_h,
                self.rel_pos_w,
                (H, W),
                (H, W),
            )
        attention_map = ops.softmax(attention_map, axis=-1)
        x = ops.reshape(
            attention_map @ values, (B, self.num_heads, H, W, self.key_dim)
        )
        x = ops.transpose(x, axes=(0, 2, 3, 1, 4))
        x = ops.reshape(x, (B, H, W, C))
        x = self.projection(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "use_bias": self.use_bias,
                "use_rel_pos": self.use_rel_pos,
                "input_size": self.input_size,
            }
        )
        return config


@keras_cv_export("keras_cv.layers.WindowedTransformerEncoder")
class WindowedTransformerEncoder(keras.layers.Layer):
    """Transformer blocks with support of window attention and residual
    propagation blocks.

    Args:
        project_dim (int): the dimensionality of the projection of the
            encoder, and output of the `MultiHeadAttention`.
        mlp_dim (int): the intermediate dimensionality of the MLP head before
            projecting to `project_dim`.
        num_heads (int): the number of heads for the `MultiHeadAttention`
            layer.
        use_bias (bool, optional): Whether to use bias to project the keys,
            queries, and values in the attention layer. Defaults to `True`.
        use_rel_pos (bool, optional): Whether to use relative positional
            emcodings in the attention layer. Defaults to `False`.
        window_size (int, optional): Window size for windowed attention.
            Defaults to `0`.
        input_size (tuple[int, int], optional): Height and width of the input
            image as a tuple of integers. Must be provided when using relative
            positional embeddings. Defaults to `None`.
        activation (str, optional): the activation function to apply in the
            MLP head - should be a function. Defaults to `"gelu"`.
        layer_norm_epsilon (float, optional): The epsilon to use in the layer
            normalization layers. Defaults to `1e-6`.
    """

    def __init__(
        self,
        project_dim,
        mlp_dim,
        num_heads,
        use_bias=True,
        use_rel_pos=False,
        window_size=0,
        input_size=None,
        activation="gelu",
        layer_norm_epsilon=1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.input_size = input_size
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.window_size = window_size
        self.use_rel_pos = use_rel_pos

        self.layer_norm1 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )
        self.layer_norm2 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )
        self.attention = MultiHeadAttentionWithRelativePE(
            num_heads=self.num_heads,
            key_dim=self.project_dim // self.num_heads,
            use_bias=use_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size
            if window_size == 0
            else (window_size, window_size),
        )
        self.mlp_block = MLPBlock(project_dim, mlp_dim, activation)

        self.layer_norm1.build([None, None, None, self.project_dim])
        self.layer_norm2.build([None, None, None, self.project_dim])

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        shortcut = x
        x = self.layer_norm1(x)
        # Window Partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]

            x, HW_padded = window_partition(x, self.window_size)

        x = self.attention(x)
        # Reverse Window Partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, HW_padded, (H, W))

        x = shortcut + x
        x = x + self.mlp_block(self.layer_norm2(x))

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "project_dim": self.project_dim,
                "mlp_dim": self.mlp_dim,
                "num_heads": self.num_heads,
                "use_bias": self.use_bias,
                "use_rel_pos": self.use_rel_pos,
                "window_size": self.window_size,
                "input_size": self.input_size,
                "activation": self.activation,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config


@keras_cv_export("keras_cv.layers.ViTDetPatchingAndEmbedding")
class ViTDetPatchingAndEmbedding(keras.layers.Layer):
    """Image to Patch Embedding using only a conv layer (without
    layer normalization).

    Args:
        kernel_size (tuple[int, int], optional): Kernel size of the
            projection layer. Defaults to `(16, 16)`.
        strides (tuple, optional): Strides of the projection layer.
            Defaults to `(16, 16)`.
        embed_dim (int, optional): Number of filters to use in the
            projection layer i.e. projection size. Defaults to `768`.
    """

    def __init__(
        self, kernel_size=(16, 16), strides=(16, 16), embed_dim=768, **kwargs
    ):
        super().__init__(**kwargs)

        self.projection = keras.layers.Conv2D(
            embed_dim, kernel_size=kernel_size, strides=strides
        )

        self.kernel_size = kernel_size
        self.strides = strides
        self.embed_dim = embed_dim

        self.built = False

    def build(self, input_shape):
        self.projection.build(input_shape)
        self.built = True

    def compute_output_shape(self, input_shape):
        return self.projection.compute_output_shape(input_shape)

    def call(self, x):
        x = self.projection(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "embed_dim": self.embed_dim,
            }
        )
        return config


def get_rel_pos(query_size, key_size, rel_pos):
    """
    Get relative positional embeddings according to the relative positions of
    query and key sizes.

    Args:
        query_size (int): The number of features of the queries.
        key_size (int): The number of features of the keys.
        rel_pos (tensor): Relative positional embedding tensor.

    Returns:
        tensor: Extracted positional embeddings according to relative
            positions.
    """
    max_rel_dist = 2 * max(query_size, key_size) - 1
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = ops.image.resize(
            images=ops.reshape(
                rel_pos, (1, rel_pos.shape[0], rel_pos.shape[1], 1)
            ),
            size=(max_rel_dist, rel_pos.shape[1]),
            interpolation="bilinear",
        )
        rel_pos_resized = ops.squeeze(rel_pos_resized, axis=(0, -1))
    else:
        rel_pos_resized = rel_pos
    query_coordinates = ops.arange(query_size, dtype="float32")[:, None] * max(
        key_size / query_size, 1.0
    )
    key_coordinates = ops.arange(key_size, dtype="float32")[None, :] * max(
        query_size / key_size, 1.0
    )
    relative_coordinates = (query_coordinates - key_coordinates) + (
        key_size - 1
    ) * max(query_size / key_size, 1.0)
    relative_coordinates = ops.cast(relative_coordinates, dtype="int64")
    return ops.take(rel_pos_resized, relative_coordinates, 0)


def add_decomposed_rel_pos(
    attention_map, queries, rel_pos_h, rel_pos_w, query_size, key_size
):
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.

    Args:
        attention_map (tensor): Attention map.
        queries (tensor): Queries in the attention layer with shape
            `(B, q_h * q_w, C)`.
        rel_pos_h (tensor): Relative position embeddings `(Lh, C)` for height
            axis.
        rel_pos_w (tensor): relative position embeddings `(Lw, C)` for width
            axis.
        query_size (tuple[int, int]): Spatial sequence size of queries with
            `(q_h, q_w)`.
        key_size (tuple[int, int]): Spatial sequence size of keys with
            `(k_h, k_w)`.

    Returns:
        tensor: attention map with added relative positional embeddings.

    References:
        - https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py  # noqa: E501
    """
    query_height, query_width = query_size
    key_height, key_width = key_size
    rel_heights = get_rel_pos(query_height, key_height, rel_pos_h)
    rel_widths = get_rel_pos(query_width, key_width, rel_pos_w)

    B, _, C = queries.shape
    rel_queries = ops.reshape(queries, (B, query_height, query_width, C))
    rel_heights = ops.einsum("bhwc,hkc->bhwk", rel_queries, rel_heights)
    rel_widths = ops.einsum("bhwc,wkc->bhwk", rel_queries, rel_widths)

    attention_map = ops.reshape(
        attention_map, (B, query_height, query_width, key_height, key_width)
    )
    attention_map = attention_map + rel_heights[..., :, None]
    attention_map = attention_map + rel_widths[..., None, :]
    attention_map = ops.reshape(
        attention_map, (B, query_height * query_width, key_height * key_width)
    )
    return attention_map


def window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_height = (window_size - H % window_size) % window_size
    pad_width = (window_size - W % window_size) % window_size
    if pad_height > 0 or pad_width > 0:
        x = ops.pad(x, ((0, 0), (0, pad_height), (0, pad_width), (0, 0)))
    H_padded, W_padded = H + pad_height, W + pad_width
    x = ops.reshape(
        x,
        (
            B,
            H_padded // window_size,
            window_size,
            W_padded // window_size,
            window_size,
            C,
        ),
    )
    windows = ops.reshape(
        ops.transpose(x, axes=(0, 1, 3, 2, 4, 5)),
        (-1, window_size, window_size, C),
    )
    return windows, (H_padded, W_padded)


def window_unpartition(windows, window_size, HW_padded, HW):
    H_padded, W_padded = HW_padded
    H, W = HW
    B = windows.shape[0] // (
        (H_padded // window_size) * (W_padded // window_size)
    )
    x = ops.reshape(
        windows,
        (
            B,
            H_padded // window_size,
            W_padded // window_size,
            window_size,
            window_size,
            -1,
        ),
    )
    x = ops.reshape(
        ops.transpose(x, axes=(0, 1, 3, 2, 4, 5)), (B, H_padded, W_padded, -1)
    )
    return x[:, :H, :W, :]
