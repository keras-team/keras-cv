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


class MLP(keras.layers.Layer):
    """A MLP block with architecture
    `input_dim -> [hidden_dim] * (num_layers - 1) -> output_dim`.

    Args:
        hidden_dim (int): The number of units in the hidden layers.
        output_dim (int): The number of units in the output layer.
        num_layers (int): The total number of dense layers to use.
        activation (str): Activation to use in the hidden layers.
            Default is `"relu"`.

    References:
        - [Segment Anything paper](https://arxiv.org/abs/2304.02643)
        - [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything)
        - [Detectron2](https://github.com/facebookresearch/detectron2)
    """  # noqa: E501

    def __init__(
        self, hidden_dim, output_dim, num_layers, activation="relu", **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation
        h = [hidden_dim] * (num_layers - 1)
        self.dense_net = []
        for hidden_dim in h:
            self.dense_net.append(keras.layers.Dense(hidden_dim))
            self.dense_net.append(keras.layers.Activation(activation))
        self.dense_net.append(keras.layers.Dense(output_dim))
        self.dense_net = keras.models.Sequential(self.dense_net)

    def build(self, input_shape):
        self.dense_net.build(input_shape)
        self.built = True

    def call(self, x):
        return self.dense_net(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
                "activation": self.activation,
            }
        )
        return config


@keras_cv_export(
    "keras_cv.layers.AddRelativePositionalEmbedding", package="keras_cv.layers"
)
class AddRelativePositionalEmbedding(keras.layers.Layer):
    def __init__(self, input_size, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.key_dim = key_dim
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
        self.built = True

    def _get_rel_pos(self, query_size, key_size, rel_pos):
        """
        Get relative positional embeddings according to the relative positions
        of query and key sizes.

        Args:
            query_size (int): The number of features of the queries.
            key_size (int): The number of features of the keys.
            rel_pos (tensor): Relative positional embedding tensor.

        Returns:
            tensor: Extracted positional embeddings according to relative
                positions.
        """
        max_rel_dist = 2 * max(query_size, key_size) - 1

        if ops.shape(rel_pos)[0] != max_rel_dist:
            rel_pos_resized = ops.image.resize(
                image=ops.reshape(
                    rel_pos,
                    (1, ops.shape(rel_pos)[0], ops.shape(rel_pos)[1], 1),
                ),
                size=(max_rel_dist, ops.shape(rel_pos)[1]),
                interpolation="bilinear",
            )
            rel_pos_resized = ops.squeeze(rel_pos_resized, axis=(0, -1))
            return rel_pos_resized
        else:
            rel_pos_resized = rel_pos
        query_coordinates = ops.cast(
            ops.arange(query_size), dtype=self.compute_dtype
        )[:, None] * (max(key_size / query_size, 1.0))
        key_coordinates = ops.cast(
            ops.arange(key_size), dtype=self.compute_dtype
        )[None, :] * (max(query_size / key_size, 1.0))
        relative_coordinates = (query_coordinates - key_coordinates) + (
            key_size - 1
        ) * max(query_size / key_size, 1.0)
        relative_coordinates = ops.cast(relative_coordinates, dtype="int32")
        return ops.take(rel_pos_resized, relative_coordinates, 0)

    def call(self, attention_map, queries, query_size, key_size):
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.

        Args:
            attention_map (tensor): Attention map.
            queries (tensor): Queries in the attention layer with shape
                `(B, q_h * q_w, C)`.
            query_size (tuple[int, int]): Spatial sequence size of queries with
                `(q_h, q_w)`.
            key_size (tuple[int, int]): Spatial sequence size of keys with
                `(k_h, k_w)`.

        Returns:
            tensor: attention map with added relative positional embeddings.

        References:
            - https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py  # noqa: E501
        """
        query_height, query_width = query_size[0], query_size[1]
        key_height, key_width = key_size[0], key_size[1]
        rel_heights = self._get_rel_pos(
            query_height, key_height, self.rel_pos_h
        )
        rel_widths = self._get_rel_pos(query_width, key_width, self.rel_pos_w)

        shape = ops.shape(queries)
        B, C = shape[0], shape[2]
        rel_queries = ops.reshape(queries, (B, query_height, query_width, C))
        rel_heights = ops.einsum("bhwc,hkc->bhwk", rel_queries, rel_heights)
        rel_widths = ops.einsum("bhwc,wkc->bhwk", rel_queries, rel_widths)

        attention_map = ops.reshape(
            attention_map, (B, query_height, query_width, key_height, key_width)
        )
        attention_map = attention_map + rel_heights[..., :, None]
        attention_map = attention_map + rel_widths[..., None, :]
        attention_map = ops.reshape(
            attention_map,
            (B, query_height * query_width, key_height * key_width),
        )
        return attention_map

    def get_config(self):
        config = super().get_config()
        config.update({"input_size": self.input_size, "key_dim": self.key_dim})
        return config


@keras_cv_export(
    "keras_cv.layers.MultiHeadAttentionWithRelativePE",
    package="keras_cv.layers",
)
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

    References:
        - [Segment Anything paper](https://arxiv.org/abs/2304.02643)
        - [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything)
        - [Detectron2](https://github.com/facebookresearch/detectron2)
    """  # noqa: E501

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
        self.input_size = input_size
        self.use_rel_pos = use_rel_pos

        self.qkv = keras.layers.Dense(
            key_dim * self.num_heads * 3, use_bias=self.use_bias
        )
        self.projection = keras.layers.Dense(key_dim * self.num_heads)

        if self.use_rel_pos:
            if input_size is None:
                raise ValueError(
                    "Input size must be provided if using relative "
                    "positional encoding."
                )
            self.add_decomposed_reative_pe = AddRelativePositionalEmbedding(
                self.input_size, self.key_dim
            )

    def build(self, input_shape=None):
        self.qkv.build([self.key_dim * self.num_heads])
        self.projection.build([self.key_dim * self.num_heads])
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        shape = ops.shape(x)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]
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
            attention_map = self.add_decomposed_reative_pe(
                attention_map,
                queries=queries,
                query_size=(H, W),
                key_size=(H, W),
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


@keras_cv_export(
    "keras_cv.layers.WindowPartitioning", package="keras_cv.layers"
)
class WindowPartitioning(keras.layers.Layer):
    def __init__(self, window_size, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.built = True

    def partition(self, x):
        shape = ops.shape(x)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]
        pad_height = (
            self.window_size - H % self.window_size
        ) % self.window_size
        pad_width = (self.window_size - W % self.window_size) % self.window_size
        if pad_height > 0 or pad_width > 0:
            x = ops.pad(x, ((0, 0), (0, pad_height), (0, pad_width), (0, 0)))
        H_padded, W_padded = H + pad_height, W + pad_width
        x = ops.reshape(
            x,
            (
                B,
                H_padded // self.window_size,
                self.window_size,
                W_padded // self.window_size,
                self.window_size,
                C,
            ),
        )
        windows = ops.reshape(
            ops.transpose(x, axes=(0, 1, 3, 2, 4, 5)),
            (-1, self.window_size, self.window_size, C),
        )
        return windows, (H_padded, W_padded)

    def unpartition(self, windows, HW_padded, HW):
        H_padded, W_padded = HW_padded
        H, W = HW
        B = ops.shape(windows)[0] // (
            (H_padded // self.window_size) * (W_padded // self.window_size)
        )
        x = ops.reshape(
            windows,
            (
                B,
                H_padded // self.window_size,
                W_padded // self.window_size,
                self.window_size,
                self.window_size,
                -1,
            ),
        )
        x = ops.reshape(
            ops.transpose(x, axes=(0, 1, 3, 2, 4, 5)),
            (B, H_padded, W_padded, -1),
        )
        return x[:, :H, :W, :]

    def get_config(self):
        config = super().get_config()
        config.update({"window_size": self.window_size})
        return config


@keras_cv_export(
    "keras_cv.layers.WindowedTransformerEncoder", package="keras_cv.layers"
)
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

    References:
        - [Segment Anything paper](https://arxiv.org/abs/2304.02643)
        - [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything)
        - [Detectron2](https://github.com/facebookresearch/detectron2)
    """  # noqa: E501

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
            input_size=(
                input_size if window_size == 0 else (window_size, window_size)
            ),
        )
        self.mlp_block = MLP(
            mlp_dim,
            project_dim,
            num_layers=2,
            activation="gelu",
        )
        self.window_partitioning = WindowPartitioning(window_size)

    def build(self, input_shape=None):
        self.layer_norm1.build([None, None, None, self.project_dim])
        self.layer_norm2.build([None, None, None, self.project_dim])
        self.attention.build()
        self.mlp_block.build([None, None, None, self.project_dim])
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        shortcut = x
        x = self.layer_norm1(x)
        # Window Partition
        if self.window_size > 0:
            H, W = ops.shape(x)[1], ops.shape(x)[2]
            x, HW_padded = self.window_partitioning.partition(x)

        x = self.attention(x)
        # Reverse Window Partition
        if self.window_size > 0:
            x = self.window_partitioning.unpartition(
                x, HW_padded=HW_padded, HW=(H, W)
            )

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


@keras_cv_export(
    "keras_cv.layers.ViTDetPatchingAndEmbedding", package="keras_cv.layers"
)
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

    References:
        - [Segment Anything paper](https://arxiv.org/abs/2304.02643)
        - [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything)
        - [Detectron2](https://github.com/facebookresearch/detectron2)
    """  # noqa: E501

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


# TODO: Merge this with the `keras_cv.layers.PatchingAndEmbedding` class once
# it has been ported to Keras Core.
@keras_cv_export(
    "keras_cv.layers.AddPositionalEmbedding", package="keras_cv.layers"
)
class AddPositionalEmbedding(keras.layers.Layer):
    def __init__(self, img_size, patch_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(
                1,
                img_size // patch_size,
                img_size // patch_size,
                embed_dim,
            ),
            initializer="zeros",
            trainable=True,
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        return x + self.pos_embed

    def get_confg(self):
        config = super().get_config()
        config.update(
            {
                "img_size": self.img_size,
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config
