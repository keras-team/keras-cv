# Copyright 2024 The KerasCV Authors
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


# ========== T5 Transformer Encoder for Stable Diffusion V3 ==========

# Adapted from KerasNLP.
# See files at: https://github.com/keras-team/keras-nlp/tree/c9baf2f4fab91832fc448624e034a30fe9a41f80/keras_nlp/models/t5  # noqa: E501

from keras_cv.backend import keras
from keras_cv.backend import ops


class T5Encoder(keras.Model):
    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        key_value_dim=None,
        dropout=0,
        use_gated_activation=True,
        layer_norm_epsilon=1e-06,
        dtype=None,
        **kwargs,
    ):
        # Token embedding layer. This layer is shared by encoder and decoder.
        self.token_embedding = keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=keras.initializers.TruncatedNormal(1.0),
            dtype=dtype,
            name="token_embedding",
        )
        self.encoder_embedding_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name="encoder_embedding_dropout",
        )
        self.encoder_transformer_layers = []
        for i in range(num_layers):
            layer = T5TransformerEncoder(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                key_value_dim=key_value_dim or hidden_dim // num_heads,
                dropout=dropout,
                layer_norm_epsilon=layer_norm_epsilon,
                num_heads=num_heads,
                use_gated_activation=use_gated_activation,
                use_relative_attention_bias=bool(i == 0),
                dtype=dtype,
                name=f"transformer_encoder_layer_{i}",
            )
            self.encoder_transformer_layers.append(layer)
        self.encoder_layer_norm = T5LayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="encoder_output_layer_norm",
        )
        self.encoder_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name="encoder_output_dropout",
        )

        # === Functional Model ===
        encoder_token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="encoder_token_ids"
        )
        encoder_padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="encoder_padding_mask"
        )

        x = self.token_embedding(encoder_token_id_input)
        x = self.encoder_embedding_dropout(x)
        encoder_attention_mask = encoder_padding_mask_input[:, None, :]
        position_bias = None
        for transformer_layer in self.encoder_transformer_layers:
            output = transformer_layer(
                x,
                attention_mask=encoder_attention_mask,
                position_bias=position_bias,
                use_causal_mask=False,
            )
            if isinstance(output, tuple):
                x, position_bias = output
        x = self.encoder_layer_norm(x)
        x = self.encoder_dropout(x)

        super().__init__(
            {
                "encoder_token_ids": encoder_token_id_input,
                "encoder_padding_mask": encoder_padding_mask_input,
            },
            outputs=x,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.key_value_dim = key_value_dim
        self.dropout = dropout
        self.use_gated_activation = use_gated_activation
        self.layer_norm_epsilon = layer_norm_epsilon

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "key_value_dim": self.key_value_dim,
                "dropout": self.dropout,
                "use_gated_activation": self.use_gated_activation,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config


class T5TransformerEncoder(keras.layers.Layer):
    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        key_value_dim,
        dropout,
        layer_norm_epsilon,
        num_heads,
        use_gated_activation=False,
        use_relative_attention_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_gated_activation = use_gated_activation

        self.self_attention = T5MultiHeadAttention(
            hidden_dim=hidden_dim,
            key_value_dim=key_value_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_relative_attention_bias=use_relative_attention_bias,
            dtype=self.dtype_policy,
            name="self_attention",
        )
        self.self_attention_layer_norm = T5LayerNorm(
            layer_norm_epsilon,
            dtype=self.dtype_policy,
        )
        self.self_attention_dropout = keras.layers.Dropout(
            dropout,
            dtype=self.dtype_policy,
        )

        self.input_projector = keras.layers.Dense(
            intermediate_dim,
            use_bias=False,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=hidden_dim**-0.5
            ),
            dtype=self.dtype_policy,
            name="input_projector",
        )
        if self.use_gated_activation:
            self.gate_projector = keras.layers.Dense(
                intermediate_dim,
                use_bias=False,
                kernel_initializer=keras.initializers.RandomNormal(
                    mean=0, stddev=hidden_dim**-0.5
                ),
                dtype=self.dtype_policy,
                name="gate_projector",
            )
        self.output_projector = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=intermediate_dim**-0.5
            ),
            dtype=self.dtype_policy,
            name="output_projector",
        )
        self.layer_norm = T5LayerNorm(
            epsilon=layer_norm_epsilon,
            dtype=self.dtype_policy,
        )
        self.dropout_layer = keras.layers.Dropout(
            dropout,
            dtype=self.dtype_policy,
        )

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        use_causal_mask=False,
        training=False,
    ):
        if use_causal_mask:
            shape = ops.shape(hidden_states)
            batch_size, length = shape[0], shape[1]
            causal_mask = compute_causal_mask(batch_size, length, length)
            attention_mask = causal_mask & ops.cast(attention_mask, "bool")

        x = hidden_states  # Intermediate result.

        residual = x
        x = self.self_attention_layer_norm(x)
        x, position_bias = self.self_attention(
            x,
            mask=attention_mask,
            position_bias=position_bias,
            training=training,
        )
        x = self.self_attention_dropout(x, training=training)
        x = x + residual

        residual = x
        x = self.layer_norm(x)
        if self.use_gated_activation:
            hidden_activation = keras.activations.gelu(
                self.input_projector(x), approximate=True
            )
            hidden_linear = self.gate_projector(x)
            x = hidden_activation * hidden_linear
        else:
            x = keras.activations.gelu(
                self.input_projector(x), approximate=True
            )
        x = self.dropout_layer(x, training=training)
        x = self.output_projector(x)
        x = self.dropout_layer(x, training=training)
        x = x + residual

        if position_bias is not None:
            return x, position_bias
        else:
            return x


class T5LayerNorm(keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=(input_shape[-1],),
            initializer="ones",
        )
        self.built = True

    def call(self, hidden_states):
        variance = ops.mean(ops.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.epsilon)
        return self.weight * hidden_states


class T5MultiHeadAttention(keras.layers.Layer):
    # This layer is adapted from Hugging Face
    # Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_tf_t5.py  # noqa: E501
    def __init__(
        self,
        hidden_dim,
        key_value_dim,
        num_heads,
        dropout,
        use_relative_attention_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.key_value_dim = key_value_dim
        self.num_heads = num_heads
        self.use_relative_attention_bias = use_relative_attention_bias

        self.inner_dim = self.num_heads * self.key_value_dim
        self.relative_attention_buckets = 32
        self.relative_attention_max_distance = 128

        self.query_projector = keras.layers.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=(self.inner_dim * self.key_value_dim) ** -0.5
            ),
            dtype=self.dtype_policy,
            name="query_projector",
        )
        self.key_projector = keras.layers.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=self.inner_dim**-0.5
            ),
            dtype=self.dtype_policy,
            name="key_projector",
        )
        self.value_projector = keras.layers.Dense(
            self.inner_dim,
            use_bias=False,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=self.inner_dim**-0.5
            ),
            dtype=self.dtype_policy,
            name="value_projector",
        )
        self.output_projector = keras.layers.Dense(
            self.hidden_dim,
            use_bias=False,
            kernel_initializer=keras.initializers.RandomNormal(
                mean=0, stddev=self.inner_dim**-0.5
            ),
            dtype=self.dtype_policy,
            name="output_projector",
        )
        self.dropout_layer = keras.layers.Dropout(
            dropout,
            dtype=self.dtype_policy,
        )

        if self.use_relative_attention_bias:
            self.relative_attention_bias = self.add_weight(
                name="embeddings",
                shape=[self.relative_attention_buckets, self.num_heads],
                initializer=keras.initializers.RandomNormal(
                    mean=0, stddev=self.inner_dim**-0.5
                ),
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """Adapted from Mesh Tensorflow.

        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position,
        i.e. the distance in tokens from the attending position to the
        attended-to position. If bidirectional=False, then positive relative
        positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute
        relative_positions. All relative positions >= max_distance map to
        the same bucket. All relative positions <= -max_distance map to
        the same bucket. This should allow for more graceful generalization to
        longer sequences than the model has been trained on.

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            Tensor with the same shape as relative_position,
            containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (
                ops.cast(
                    ops.greater(relative_position, 0),
                    dtype=relative_position.dtype,
                )
                * num_buckets
            )
            relative_position = ops.abs(relative_position)
        else:
            relative_position = -ops.minimum(relative_position, 0)
        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = ops.less(relative_position, max_exact)
        relative_position_if_large = max_exact + ops.cast(
            ops.log(
                ops.cast(relative_position, "float32")
                / ops.cast(max_exact, "float32")
            )
            / ops.cast(ops.log(max_distance / max_exact), "float32")
            * (num_buckets - max_exact),
            dtype=relative_position.dtype,
        )
        relative_position_if_large = ops.minimum(
            relative_position_if_large, num_buckets - 1
        )
        relative_buckets += ops.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = ops.arange(query_length)[:, None]
        memory_position = ops.arange(key_length)[None, :]
        relative_position = (
            memory_position - context_position
        )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.relative_attention_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = ops.take(
            self.relative_attention_bias, relative_position_bucket, axis=0
        )  # shape (query_length, key_length, num_heads)
        values = ops.expand_dims(
            ops.transpose(values, axes=(2, 0, 1)), axis=0
        )  # shape (1, num_heads, query_length, key_length)
        return values

    def call(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        training=False,
    ):
        # Input is (batch_size, query_length, dim)
        # past_key_value[0] is (batch_size, num_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = ops.shape(hidden_states)[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"Argument `past_key_value` should have 2 past states: "
                    f"keys and values. Got {len(past_key_value)} past states."
                )
            real_seq_length += (
                ops.shape(past_key_value[0])[2]
                if query_length is None
                else query_length
            )

        key_length = (
            real_seq_length
            if key_value_states is None
            else ops.shape(key_value_states)[1]
        )

        def shape(hidden_states):
            return ops.transpose(
                ops.reshape(
                    hidden_states,
                    (batch_size, -1, self.num_heads, self.key_value_dim),
                ),
                axes=(0, 2, 1, 3),
            )

        def unshape(hidden_states):
            return ops.reshape(
                ops.transpose(hidden_states, axes=(0, 2, 1, 3)),
                (batch_size, -1, self.inner_dim),
            )

        def project(
            hidden_states, proj_layer, key_value_states, past_key_value
        ):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attention
                # (batch_size, num_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attention
                # (batch_size, num_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attention
                    # (batch_size, num_heads, key_length, dim_per_head)
                    hidden_states = ops.concat(
                        [past_key_value, hidden_states], axis=2
                    )
                else:
                    # cross-attention
                    hidden_states = past_key_value
            return hidden_states

        # get query
        query_states = shape(
            self.query_projector(hidden_states)
        )  # (batch_size, num_heads, query_length, dim_per_head)

        # get key/value
        key_states = project(
            hidden_states,
            self.key_projector,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            self.value_projector,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        scores = ops.einsum(
            "bnqd,bnkd->bnqk", query_states, key_states
        )  # (batch_size, num_heads, query_length, key_length)

        if position_bias is None:
            if not self.use_relative_attention_bias:
                position_bias = ops.zeros(
                    (1, self.num_heads, real_seq_length, key_length),
                    self.compute_dtype,
                )
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated we want only
            # the last query position bias
            if past_key_value is not None:
                if not self.use_relative_attention_bias:
                    position_bias = position_bias[:, :, -seq_length:, :]
                else:
                    # we might have a padded past structure,
                    # in which case we want to fetch the position bias slice
                    # right after the most recently filled past index
                    most_recently_filled_past_index = ops.amax(
                        ops.where(past_key_value[0][0, 0, :, 0] != 0.0)
                    )
                    position_bias = ops.slice(
                        position_bias,
                        (0, 0, most_recently_filled_past_index + 1, 0),
                        (1, self.num_heads, seq_length, real_seq_length),
                    )

            if mask is not None:
                # Add a new mask axis for the head dim.
                mask = mask[:, None, :, :]
                # Add a very large negative position bias for masked positions.
                mask = (1.0 - ops.cast(mask, position_bias.dtype)) * -1e9
                position_bias = position_bias + mask

        scores += ops.cast(position_bias, scores.dtype)
        weights = ops.nn.softmax(
            scores, axis=-1
        )  # (batch_size, num_heads, query_length, key_length)
        weights = self.dropout_layer(
            weights, training=training
        )  # (batch_size, num_heads, query_length, key_length)

        # Optionally mask heads
        if layer_head_mask is not None:
            weights = ops.reshape(layer_head_mask, (1, -1, 1, 1)) * weights

        attention_output = ops.matmul(
            weights, value_states
        )  # (batch_size, num_heads, query_length, dim_per_head)

        attention_output = self.output_projector(unshape(attention_output))
        return (attention_output, position_bias)


def compute_causal_mask(batch_size, input_length, output_length, cache_index=0):
    """Compute a causal attention mask for a transformer decoder.

    Args:
        batch_size: batch size for the mask.
        input_length: the length of key/value tensors in the attention layer.
        output_length: the length of query tensors in the attention layer.
        cache_index: the current index for cached generation. If passed, the
            query sequence will be considered to start at `cache_index` rather
            than zero. For example, a causal mask with `output_length=1` and
            `cache_index=5` would allow the query tensor to attend to the first
            five positions of the key/value tensors.

    Return:
        A causal attention mask with shape
        `(batch_size, output_length, input_length)` that can be passed to a
        attention layer.
    """
    i = ops.arange(output_length, dtype="float32")
    i = i + ops.cast(cache_index, "float32")
    i = ops.expand_dims(i, axis=1)
    j = ops.arange(input_length, dtype="float32")
    mask = ops.expand_dims(i >= j, axis=0)

    return ops.broadcast_to(mask, (batch_size, output_length, input_length))
