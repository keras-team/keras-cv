import string

import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras import layers

from keras_cv.layers.mbconv import MBConvBlock
from keras_cv.layers.transformer_encoder import MaxViTTransformerEncoder

_CACHE = {}


def maybe_reset_cache():
    """Resets the constants if the default graph changes."""
    global _CACHE

    def _get_tensor(t):
        if isinstance(t, tf.Tensor):
            return t
        elif isinstance(t, dict):
            return _get_tensor(list(t.values())[0])
        elif isinstance(t, (tuple, list)):
            return _get_tensor(t[0])
        else:
            raise ValueError("Unsupport cache type %d" % type(t))

    if _CACHE:
        _CACHE = {}


def generate_lookup_tensor(
    length, max_relative_position=None, clamp_out_of_range=False, dtype=tf.float32
):
    """Generate a one_hot lookup tensor to reindex embeddings along one dimension.

    Args:
        length: the length to reindex to.
        max_relative_position: the maximum relative position to consider.
        Relative position embeddings for distances above this threshold
        are zeroed out.
        clamp_out_of_range: bool. Whether to clamp out of range locations to the
        maximum relative distance. If False, the out of range locations will be
        filled with all-zero vectors.
        dtype: dtype for the returned lookup tensor.
    Returns:
        a lookup Tensor of size [length, length, vocab_size] that satisfies
        ret[n,m,v] = 1{m - n + max_relative_position = v}.
    """
    maybe_reset_cache()
    if max_relative_position is None:
        max_relative_position = length - 1
    lookup_key = ("lookup_matrix", length, max_relative_position)
    # Return the cached lookup tensor, otherwise compute it and cache it.
    if lookup_key not in _CACHE:
        vocab_size = 2 * max_relative_position + 1
        ret = np.zeros((length, length, vocab_size))
        for i in range(length):
            for x in range(length):
                v = x - i + max_relative_position
                if abs(x - i) > max_relative_position:
                    if clamp_out_of_range:
                        v = np.clip(v, 0, vocab_size - 1)
                    else:
                        continue
                ret[i, x, v] = 1
        _CACHE[lookup_key] = tf.constant(ret, dtype)
    return _CACHE[lookup_key]


def reindex_2d_einsum_lookup(
    relative_position_tensor,
    height,
    width,
    max_relative_height=None,
    max_relative_width=None,
    h_axis=None,
):
    """Reindex 2d relative position bias with 2 independent einsum lookups.

    Args:
        relative_position_tensor: tensor of shape
        [..., vocab_height, vocab_width, ...].
        height: height to reindex to.
        width: width to reindex to.
        max_relative_height: maximum relative height.
        Position embeddings corresponding to vertical distances larger
        than max_relative_height are zeroed out. None to disable.
        max_relative_width: maximum relative width.
        Position embeddings corresponding to horizontal distances larger
        than max_relative_width are zeroed out. None to disable.
        h_axis: Axis corresponding to vocab_height. Default to 0 if None.

    Returns:
        reindexed_tensor: a Tensor of shape
        [..., height * width, height * width, ...]
    """
    height_lookup = generate_lookup_tensor(
        height,
        max_relative_position=max_relative_height,
        dtype=relative_position_tensor.dtype,
    )
    width_lookup = generate_lookup_tensor(
        width,
        max_relative_position=max_relative_width,
        dtype=relative_position_tensor.dtype,
    )

    if h_axis is None:
        h_axis = 0

    non_spatial_rank = relative_position_tensor.shape.rank - 2
    non_spatial_expr = "".join(chr(ord("n") + i) for i in range(non_spatial_rank))
    prefix = non_spatial_expr[:h_axis]
    suffix = non_spatial_expr[h_axis:]

    reindexed_tensor = tf.einsum(
        "{0}hw{1},ixh->{0}ixw{1}".format(prefix, suffix),
        relative_position_tensor,
        height_lookup,
        name="height_lookup",
    )
    reindexed_tensor = tf.einsum(
        "{0}ixw{1},jyw->{0}ijxy{1}".format(prefix, suffix),
        reindexed_tensor,
        width_lookup,
        name="width_lookup",
    )

    ret_shape = relative_position_tensor.shape.as_list()
    ret_shape[h_axis] = height * width
    ret_shape[h_axis + 1] = height * width
    reindexed_tensor = tf.reshape(reindexed_tensor, ret_shape)

    return reindexed_tensor


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class WindowPartitioning(layers.Layer):
    """
    Based on: https://github.com/google-research/maxvit/blob/2e06a7f1f70c76e64cd3dabe5cd1b8c1a23c9fb7/maxvit/models/maxvit.py#L805
    Partition the input feature maps into non-overlapping windows.

        Args:
          inputs: [B, H, W, C] feature maps.
        Returns:
          A `tf.Tensor`: Partitioned features: [B, nH, nW, wSize, wSize, c].
        Raises:
          ValueError: If the feature map sizes are not divisible by window sizes.


    Basic usage:

    ```
    inputs = tf.random.normal([1, 256, 256, 3])

    layer = keras_cv.layers.WindowPartitioning(window_size=64)
    outputs = layer(inputs)
    outputs.shape # TensorShape([16, 64, 64, 3])
    ```
    """

    def __init__(self, window_size, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size

    def call(self, input):
        _, h, w, c = input.shape
        window_size = self.window_size

        if h % window_size != 0 or w % window_size != 0:
            raise ValueError(
                f"Feature map sizes {(h, w)} "
                f"not divisible by window size ({window_size})."
            )

        features = tf.reshape(
            input, (-1, h // window_size, window_size, w // window_size, window_size, c)
        )
        features = tf.transpose(features, (0, 1, 3, 2, 4, 5))
        features = tf.reshape(features, (-1, window_size, window_size, c))
        return features

    def get_config(self):
        config = {"window_size": self.window_size}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class UnWindowPartitioning(layers.Layer):
    """
    Based on: https://github.com/google-research/maxvit/blob/2e06a7f1f70c76e64cd3dabe5cd1b8c1a23c9fb7/maxvit/models/maxvit.py#L832
    Reverses the operation of the WindowPartitioning layer.

        Args:
          window_size: the window size used while performing WindowPartitioning
          height: the desired height to stitch the feature map to
          width: the desired width to stitch the feature map to
        Returns:
          A `tf.Tensor`: Stiched feature map: [B, H, W, C].

    Basic usage:

    ```
    inputs = tf.random.normal([1, 256, 256, 3])

    window_layer = keras_cv.layers.WindowPartitioning(window_size=64)
    outputs = window_layer(inputs)

    unwindow_layer = keras_cv.layers.UnWindowPartitioning(window_size=64, height=256, width=256)
    unwindow_outputs = unwindow_layer(outputs)
    unwindow_outputs.shape # TensorShape([1, 256, 256, 3])

    import numpy as np
    np.testing.assert_array_equal(unwindow_outputs, inputs)
    ```
    """

    def __init__(self, window_size, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size

    def call(self, input, height, width):
        features = tf.reshape(
            input,
            [
                -1,
                height // self.window_size,
                width // self.window_size,
                self.window_size,
                self.window_size,
                input.shape[-1],
            ],
        )
        return tf.reshape(
            tf.transpose(features, (0, 1, 3, 2, 4, 5)),
            [-1, height, width, features.shape[-1]],
        )

    def get_config(self):
        config = {
            "window_size": self.window_size,
            "height": self.height,
            "width": self.width,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class GridPartitioning(layers.Layer):
    """
    Based on: https://github.com/google-research/maxvit/blob/2e06a7f1f70c76e64cd3dabe5cd1b8c1a23c9fb7/maxvit/models/maxvit.py#L842
    Partition the input feature maps into non-overlapping windows.
        Args:
          features: [B, H, W, C] feature maps.
        Returns:
          Partitioned features: [B, nH, nW, wSize, wSize, c].
        Raises:
          ValueError: If the feature map sizes are not divisible by window sizes.

    Basic usage:

    ```
    inputs = tf.random.normal([1, 256, 256, 3])

    layer = keras_cv.layers.GridPartitioning(grid_size=64)
    outputs = layer(inputs)
    outputs.shape # TensorShape([16, 64, 64, 3])
    ```
    """

    def __init__(self, grid_size, **kwargs):
        super().__init__(**kwargs)
        self.grid_size = grid_size

    def call(self, input):
        _, h, w, c = input.shape
        grid_size = self.grid_size
        if h % grid_size != 0 or w % grid_size != 0:
            raise ValueError(
                f"Feature map sizes {(h, w)} "
                f"not divisible by window size ({grid_size})."
            )
        features = tf.reshape(
            input, (-1, grid_size, h // grid_size, grid_size, w // grid_size, c)
        )
        features = tf.transpose(features, (0, 2, 4, 1, 3, 5))
        features = tf.reshape(features, (-1, grid_size, grid_size, c))
        return features

    def get_config(self):
        config = {"grid_size": self.grid_size}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class UnGridPartitioning(layers.Layer):
    """
    Based on: https://github.com/google-research/maxvit/blob/2e06a7f1f70c76e64cd3dabe5cd1b8c1a23c9fb7/maxvit/models/maxvit.py#L867
    Reverses the operation of the GridPartitioning layer.

        Args:
          grid_size: the grid size used while performing GridPartitioning
          height: the desired height to stitch the feature map to
          width: the desired width to stitch the feature map to
        Returns:
          A `tf.Tensor`: Stiched feature map: [B, H, W, C].

    Basic usage:

    ```
    inputs = tf.random.normal([1, 256, 256, 3])

    grid_layer = keras_cv.layers.GridPartitioning(grid_size=64)
    outputs = grid_layer(inputs)

    ungrid_layer = keras_cv.layers.UnGridPartitioning(grid_size=64, height=256, width=256)
    ungrid_outputs = ungrid_layer(outputs)
    ungrid_outputs.shape # TensorShape([1, 256, 256, 3])

    import numpy as np
    np.testing.assert_array_equal(ungrid_outputs, inputs)
    ```
    """

    def __init__(self, grid_size, **kwargs):
        super().__init__(**kwargs)
        self.grid_size = grid_size

    def call(self, input, height, width):
        features = tf.reshape(
            input,
            [
                -1,
                height // self.grid_size,
                width // self.grid_size,
                self.grid_size,
                self.grid_size,
                input.shape[-1],
            ],
        )
        return tf.reshape(
            tf.transpose(features, (0, 3, 1, 4, 2, 5)),
            [-1, height, width, features.shape[-1]],
        )

    def get_config(self):
        config = {
            "grid_size": self.grid_size,
            "height": self.height,
            "width": self.width,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class MaxViTStem(layers.Layer):
    # Conv blocks
    def __init__(
        self,
        filters: list = [64, 64],
        kernel_size: tuple = (3, 3),
        kernel_initializer=tf.random_normal_initializer(stddev=0.02),
        bias_initializer=tf.zeros_initializer,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert len(filters) == 2

        self.conv1 = layers.Conv2D(
            filters=filters[0],
            kernel_size=kernel_size,
            strides=2,
            padding="same",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="stem_conv_0",
        )

        self.batch_norm = layers.BatchNormalization()
        self.gelu = layers.Activation("gelu")

        self.conv2 = layers.Conv2D(
            filters=filters[1],
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name="stem_conv_1",
        )

    def call(self, input):
        # Creates a stem for the MaxViT model.
        x = self.conv1(input)
        x = self.batch_norm(x)
        x = self.gelu(x)
        x = self.conv2(x)
        return x

    def get_config(self):
        # config = {"...": self....}
        # base_config = super().get_config()
        # return dict(list(base_config.items()) + list(config.items()))
        return super().get_config()

    def get_config(self):
        # config = {"...": self....}
        # base_config = super().get_config()
        # return dict(list(base_config.items()) + list(config.items()))
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RelativeMultiHeadAttention(layers.MultiHeadAttention):
    def __init__(
        self,
        num_heads,
        key_dim,
        value_dim=None,
        scale_ratio=None,
        dropout=0.0,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    ):
        super().__init__(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            dropout=dropout,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )

        self._num_heads = num_heads
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._scale_ratio = scale_ratio

    def build(self, query_shape, value_shape=None, key=None):
        query_shape_list = query_shape.as_list()
        if query_shape.rank == 4:
            height, width = query_shape_list[1:3]
        elif query_shape.rank == 3:
            seq_len = query_shape_list[1]
            height = int(seq_len**0.5)
            width = height
            if height * width != seq_len:
                raise ValueError(
                    "Does not support 2D relative attentive for " "non-square inputs."
                )
        else:
            raise ValueError(
                "Does not support relative attention for query shape: %s."
                % query_shape_list
            )

        if self._scale_ratio is not None:
            scale_ratio = eval(self._scale_ratio)
            vocab_height = 2 * round(height / scale_ratio) - 1
            vocab_width = 2 * round(width / scale_ratio) - 1
        else:
            vocab_height = 2 * height - 1
            vocab_width = 2 * width - 1

        relative_bias_shape = [self._num_heads, vocab_height, vocab_width]

        self.relative_bias = self.add_weight(
            "relative_bias",
            relative_bias_shape,
            initializer=self._kernel_initializer,
            trainable=True,
        )

        if self._scale_ratio is not None:
            src_shape = self.relative_bias.shape.as_list()
            relative_bias = tf.expand_dims(self.relative_bias, axis=-1)
            relative_bias = tf.cast(
                tf.image.resize(relative_bias, [2 * height - 1, 2 * width - 1]),
                self.compute_dtype,
            )
            relative_bias = tf.squeeze(relative_bias, axis=-1)
            tgt_shape = relative_bias.shape.as_list()
        else:
            relative_bias = tf.cast(self.relative_bias, self.compute_dtype)

        self.reindexed_bias = reindex_2d_einsum_lookup(
            relative_bias, height, width, height - 1, width - 1, h_axis=1
        )

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        """Applies Dot-product attention with query, key, value tensors.
        This function defines the computation inside `call` with projected
        multi-head Q, K, V inputs.
        Args:
          query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
          key: Projected key `Tensor` of shape `(B, S, N, key_dim)`.
          value: Projected value `Tensor` of shape `(B, S, N, value_dim)`.
          attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
            attention to certain positions. It is generally not needed if the
            `query` and `value` (and/or `key`) are masked.
          training: Python boolean indicating whether the layer should behave in
            training mode (adding dropout) or in inference mode (doing nothing).
        Returns:
          attention_output: Multi-headed outputs of attention computation.
          attention_scores: Multi-headed attention weights.
        """
        # Note: Applying scalar multiply at the smaller end of einsum improves
        # XLA performance, but may introduce slight numeric differences in
        # the Transformer attention head.
        query = tf.multiply(query, 1.0 / float(self._key_dim) ** 0.5)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = tf.einsum(self._dot_product_equation, key, query)

        # Add relative bias
        if self.reindexed_bias is not None:
            attention_scores += self.reindexed_bias

        attention_scores = self._masked_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )

        # `context_layer` = [B, T, N, H]
        attention_output = tf.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores

    def call(
        self,
        query,
        value,
        key=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
    ):
        return super().call(
            query,
            value,
            key,
            attention_mask,
            return_attention_scores,
            training,
            use_causal_mask,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self._num_heads,
                "scale_ratio": self._scale_ratio,
                "kernel_initializer": initializers.serialize(self._kernel_initializer),
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class MaxViTBlock(layers.Layer):

    """
    Performs MBConv -> Block-Attention (Block-SA+FFN) -> Grid-Attention (Grid-SA+FFN))

    The repeated MaxViT block that passes the input through the MBConv block, Block-Attention with an FFN top
    and Grid-Attention with an FFN top.
    Within Block-Attention and Grid-Attention, the input is passed through a window/grid reshaping operation,
    flattened to a 1 dimension, passed through the RelativeMultiHeadAttention layer, and then reshaped back.

        Args:
            hidden_size: the hidden size to be used in the MBConvBlock, FFN heads and RelativeMultiHeadAttention layers,
            head_size: the head size for RelativeMultiHeadAttention layers,
            window_size: the window_size to be used for WindowPartition and UnWindowPartition,
            grid_size: the grid_size to be used for GridPartition and UnGridPartition,
            dropout: default None, the dropout to apply after attention and before adding the residual,
            num_heads: default None, the number of heads to use in RelativeMultiHeadAttention, computed as
                self.hidden_size // self.head_size if None.
            dropatt: default None, the dropout to apply in RelativeMultiHeadAttention
            rel_attn_type: default "2d_multi_head", the type of RelativeMultiHeadAttention to use
            expansion_rate: default 4, the expansion rate for EinsumDense layers in the FFN heads
            activation: default "gelu", the activation function to apply in the FFN heads
            pool_type: default "avg", the pooling to use in MBConvBlocks
            pool_stride: default 1, the strides to use in MBConvBlocks
            scale_ratio: default None,
            survival_prob: default None, the survival_probability for the MBConvBlock
            ln_epsilon: default 1e-5, the layer normalization epsilon
            ln_dtype: default None, the layer normalization dtype
            kernel_initializer: default tf.random_normal_initializer(stddev=0.02), the kernel_initializer for the FFN head
            bias_initializer: default tf.zeros_initializer, the bias initializer for the FFN head
        Returns:
          A `tf.Tensor` resulting from the operations above.

    Basic usage:

    ```
    # Start of MaxViT
    inputs = input_img = tf.random.uniform((1, 224, 224, 3), minval=0, maxval=1)

    stem = keras_cv.layers.MaxViTStem()
    stem_out = stem(input_img) # TensorShape([1, 112, 112, 64])

    maxvit_block = keras_cv.layers.MaxViTBlock(hidden_size=64,
                                                head_size=32,
                                                window_size=7,
                                                grid_size=7,
                                                pool_stride=2)
    block_out = maxvit_block(stem_out)
    block_out.shape # TensorShape([1, 56, 56, 64])

    # Later blocks don't use stride

    maxvit_block = keras_cv.layers.MaxViTBlock(hidden_size=64,
                                                head_size=32,
                                                window_size=7,
                                                grid_size=7)
    block_out = maxvit_block(block_out)
    block_out.shape # TensorShape([1, 56, 56, 64])
    ```
    """

    def __init__(
        self,
        hidden_size,
        head_size,
        window_size,
        grid_size,
        dropout=None,
        num_heads=None,
        expansion_rate=4,
        activation="gelu",
        pool_type="avg",
        pool_stride=1,
        dropatt=None,
        rel_attn_type="2d_multi_head",
        scale_ratio=None,
        survival_prob=None,
        ln_epsilon=1e-5,
        ln_dtype=None,
        kernel_initializer=tf.random_normal_initializer(stddev=0.02),
        bias_initializer=tf.zeros_initializer,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.head_size = head_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.grid_size = grid_size
        self.num_heads = num_heads
        self.expansion_rate = expansion_rate
        self.activation = activation
        self.pool_type = pool_type
        self.pool_stride = pool_stride
        self.dropatt = dropatt
        self.rel_attn_type = rel_attn_type
        self.scale_ratio = scale_ratio
        self.survival_prob = survival_prob
        self.ln_epsilon = ln_epsilon
        self.ln_dtype = ln_dtype
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.mbconv = MBConvBlock(
            input_filters=self.hidden_size,
            output_filters=self.hidden_size,
            strides=self.pool_stride,
            expand_ratio=self.expansion_rate,
            survival_probability=self.survival_prob,
            se_ratio=0.25,
        )
        self.transformer_encoder = MaxViTTransformerEncoder(
            hidden_size,
            head_size=self.head_size,
            window_size=self.window_size,
            grid_size=self.grid_size,
            dropout=self.dropout,
            num_heads=self.num_heads,
            expansion_rate=self.expansion_rate,
            activation=self.activation,
            dropatt=self.dropatt,
            rel_attn_type=self.rel_attn_type,
            scale_ratio=self.scale_ratio,
            ln_epsilon=self.ln_epsilon,
            ln_dtype=self.ln_dtype,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )

    def call(self, input):
        # MBConv
        x = self.mbconv(input)
        x = self.transformer_encoder(x)
        return x

    def get_config(self):
        config = {
            "dropout": self.dropout,
            "head_size": self.head_size,
            "hidden_size": self.hidden_size,
            "window_size": self.window_size,
            "grid_size": self.grid_size,
            "num_heads": self.num_heads,
            "expansion_rate": self.expansion_rate,
            "activation": self.activation,
            "pool_type": self.pool_type,
            "pool_stride": self.pool_stride,
            "dropatt": self.dropatt,
            "rel_attn_type": self.rel_attn_type,
            "scale_ratio": self.scale_ratio,
            "survival_prob": self.survival_prob,
            "ln_epsilon": self.ln_epsilon,
            "ln_dtype": self.ln_dtype,
            "kernel_initializer": tf.keras.initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class _FFN:
    def __init__(
        self,
        hidden_size,
        dropout=0.0,
        expansion_rate=4,
        activation="gelu",
        kernel_initializer=tf.random_normal_initializer(stddev=0.02),
        bias_initializer=tf.zeros_initializer,
        name="ffn",
    ):
        self.hidden_size = hidden_size
        self.expansion_rate = expansion_rate
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.expanded_size = self.hidden_size * self.expansion_rate
        self.dropout = dropout
        self.activation = layers.Activation(activation)

    def build(self, input_shape):
        input_rank = input_shape.rank
        shared_size = -1 % input_rank
        i_only_size = input_rank - shared_size
        o_only_size = len(self.hidden_size)

        assert input_rank + o_only_size < len(
            string.ascii_uppercase
        ), "Cannot use einsum as input rank + output rank > 26."
        einsum_str = string.ascii_uppercase[: input_rank + o_only_size]

        offset = 0
        shared_str = einsum_str[offset : offset + shared_size]
        offset += shared_size
        i_only_str = einsum_str[offset : offset + i_only_size]
        offset += i_only_size
        o_only_str = einsum_str[offset : offset + o_only_size]

        input_str = "{}{}".format(shared_str, i_only_str)
        output_str = "{}{}".format(shared_str, o_only_str)
        weight_str = "{}{}".format(i_only_str, o_only_str)

        self.einsum_expr = "{},{}->{}".format(input_str, weight_str, output_str)

        self._expand_dense = layers.EinsumDense(
            equation=self.einsum_expr,
            output_shape=self.expanded_size,
            output_trailing_dims=self.expanded_size,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="expand_dense",
        )
        self._shrink_dense = layers.EinsumDense(
            equation=self.einsum_expr,
            output_shape=self.hidden_size,
            output_trailing_dims=self.hidden_size,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name="shrink_dense",
        )

    def call(self, input):
        output = input
        output = self._expand_dense(output)
        output = self.activation(output)
        if self.dropout:
            output = tf.keras.layers.Dropout(self.dropout, name="nonlinearity_drop")(
                output
            )
        output = self._shrink_dense(output)

        return output

    def get_config(self):
        config = {
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "expansion_rate": self.expansion_rate,
            "activation": tf.keras.activations.serialize(self.activation),
            "kernel_initializer": tf.keras.initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
