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


def get_initializer(initializer_range=0.02):
    """
    Creates a `keras.initializers.TruncatedNormal` with the given range.

    Args:
        initializer_range (*float*, defaults to 0.02): Standard deviation of the initializer range.

    Returns:
        `keras.initializers.TruncatedNormal`: The truncated normal initializer.
    """
    return keras.initializers.TruncatedNormal(stddev=initializer_range)


class QuickGELU(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return x * ops.sigmoid(1.702 * x)


class ResidualAttention(keras.layers.Layer):
    def __init__(
        self,
        proj_dim,
        n_head,
        num_hidden_layers,
        attn_mask=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.proj_dim = proj_dim
        self.n_head = n_head
        self.attn_mask = attn_mask
        self.num_hidden_layers = num_hidden_layers
        self.fc_std = ops.power(2 * self.proj_dim, -0.5) * 0.02

        self.in_proj_std = (
            ops.power(self.proj_dim, -0.5)
            * (ops.power(2 * self.num_hidden_layers, -0.5))
            * 0.02
        )

    def attention(self, x, attention_mask=None):
        mask = (
            ops.cast(self.attn_mask, dtype=x.dtype)
            if self.attn_mask is not None
            else None
        )
        if attention_mask is not None:
            attention_mask = (
                ops.cast(attention_mask, dtype=x.dtype)
                if attention_mask is not None
                else None
            )
            mask = ops.add(self.attn_mask, attention_mask)

        return self.attn(
            x,
            attention_mask=mask,
        )

    def build(self, input_shape):
        super().build(input_shape)
        self.attn = CLIPAttention(
            self.proj_dim,
            self.n_head,
            self.num_hidden_layers,
            name="multi_head_attention",
        )
        self.ln_1 = keras.layers.LayerNormalization(epsilon=1e-5, name="ln_1")
        self.mlp = keras.Sequential(
            [
                keras.layers.Dense(
                    self.proj_dim * 4,
                    kernel_initializer=get_initializer(self.in_proj_std),
                    name="c_fc",
                ),
                QuickGELU(name="gelu"),
                keras.layers.Dense(
                    self.proj_dim,
                    kernel_initializer=get_initializer(self.fc_std),
                    name="c_proj",
                ),
            ]
        )
        self.ln_2 = keras.layers.LayerNormalization(epsilon=1e-5, name="ln_2")

    def call(self, x, attention_mask=None):
        x = x + self.attention(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

    def compute_output_shape(self, inputs_shape):
        return inputs_shape


class CLIPEncoder(keras.layers.Layer):
    def __init__(self, width, layers, heads, attn_mask=None, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.layers = layers
        self.heads = heads
        self.attn_mask = attn_mask
        self.resblocks = [
            ResidualAttention(
                self.width, self.heads, self.layers, self.attn_mask
            )
            for _ in range(self.layers)
        ]

    def build(self, input_shape):
        super().build(input_shape)
        self.resblocks.build()

    def call(self, x, attention_mask=None):
        for block in self.resblocks:
            x = block(x, attention_mask=attention_mask)
        return x

    def compute_output_shape(self, inputs_shape):
        return inputs_shape


class CLIPAttention(keras.layers.Layer):
    """
    - Documentation page: https://huggingface.co/docs/transformers/model_doc/clip # noqa: E501
    - Implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py # noqa: E501
    """

    def __init__(
        self, project_dim, num_heads, num_hidden_layers, dropout=0.0, **kwargs
    ):
        super().__init__(**kwargs)

        self.project_dim = project_dim
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers
        self.head_dim = self.project_dim // self.num_heads
        if self.head_dim * self.num_heads != self.project_dim:
            raise ValueError(
                f"project_dim must be divisible by num_heads (got `project_dim`"
                f": {self.project_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.sqrt_att_head_size = ops.sqrt(self.head_dim)
        self.scale = self.head_dim**-0.5
        in_proj_std = (
            (self.project_dim**-0.5)
            * ((2 * self.num_hidden_layers) ** -0.5)
            * 0.02
        )
        out_proj_std = (self.project_dim**-0.5) * 0.02
        self.dropout = dropout
        self.q_proj = keras.layers.Dense(
            units=self.project_dim,
            kernel_initializer=get_initializer(in_proj_std),
            name="q_proj",
        )
        self.k_proj = keras.layers.Dense(
            units=self.project_dim,
            kernel_initializer=get_initializer(in_proj_std),
            name="k_proj",
        )
        self.v_proj = keras.layers.Dense(
            units=self.project_dim,
            kernel_initializer=get_initializer(in_proj_std),
            name="v_proj",
        )
        self.out_proj = keras.layers.Dense(
            units=self.project_dim,
            kernel_initializer=get_initializer(out_proj_std),
            name="out_proj",
        )

    def build(self, input_shape):
        super().build(input_shape)
        self.q_proj.build(input_shape)
        self.k_proj.build(input_shape)
        self.v_proj.build(input_shape)
        self.out_proj.build(input_shape)

    def _transpose_for_scores(self, tensor, batch_size):
        """
        Copied from https://github.com/huggingface/transformers/blob/8e164c5400b7b413c7b8fb32e35132001effc970/src/transformers/models/bert/modeling_tf_bert.py#L252 # noqa: E501
        """
        # [batch_size, seq_len, all_head_dim] ->
        # [batch_size, seq_len, num_heads, head_dim]
        tensor = ops.reshape(
            tensor, (batch_size, -1, self.num_heads, self.head_dim)
        )
        # [batch_size, seq_len, num_heads, head_dim] ->
        # [batch_size, num_heads, seq_len, head_dim]
        return ops.transpose(tensor, axes=[0, 2, 1, 3])

    def call(
        self,
        x,
        attention_mask=None,
        causal_attention_mask=None,
        output_attentions=None,
        training=False,
    ):
        batch_size = ops.shape(x)[0]
        mixed_query_layer = self.q_proj(inputs=x)
        mixed_key_layer = self.k_proj(inputs=x)
        mixed_value_layer = self.v_proj(inputs=x)
        query_layer = self._transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self._transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self._transpose_for_scores(mixed_value_layer, batch_size)

        # Scaled dot product between key and query = raw attention scores.
        attention_scores = ops.matmul(
            query_layer, ops.transpose(key_layer, axes=[0, 1, 3, 2])
        )
        dk = ops.cast(ops.sqrt(self.head_dim), dtype=attention_scores.dtype)
        attention_scores = ops.divide(
            attention_scores, dk
        )  # (batch_size, num_heads, seq_len_q, seq_len_k)

        # Apply the causal_attention_mask first
        if causal_attention_mask is not None:
            # Apply the causal attention mask (precomputed for all layers in
            # the call() function)
            attention_scores = ops.add(attention_scores, causal_attention_mask)

        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in the
            # call() function)
            attention_scores = ops.add(attention_scores, attention_mask)

        # Normalize the attention scores to probabilities.
        _attention_probs = ops.softmax(attention_scores + 1e-9, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = keras.layers.Dropout(self.dropout)(
            inputs=_attention_probs, training=training
        )

        attn_output = ops.matmul(attention_probs, value_layer)
        attn_output = ops.transpose(attn_output, axes=[0, 2, 1, 3])

        # (batch_size, seq_len_q, project_dim)
        attn_output = ops.reshape(
            attn_output, (batch_size, -1, self.project_dim)
        )

        attn_output = self.out_proj(attn_output, training=training)
        outputs = (
            (attn_output, _attention_probs)
            if output_attentions
            else attn_output
        )

        return outputs
