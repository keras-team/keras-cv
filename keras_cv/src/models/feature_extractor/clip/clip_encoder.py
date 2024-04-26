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


@keras_cv_export("keras_cv.models.feature_extractor.QuickGELU")
class QuickGELU(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return x * ops.sigmoid(1.702 * x)


@keras_cv_export("keras_cv.models.feature_extractor.ResidualAttention")
class ResidualAttention(keras.layers.Layer):
    def __init__(
        self,
        proj_dim,
        num_heads,
        num_hidden_layers,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers
        self.attn = CLIPAttention(
            self.proj_dim,
            self.num_heads,
            self.num_hidden_layers,
            name="multi_head_attention",
        )
        self.ln_1 = keras.layers.LayerNormalization(epsilon=1e-5, name="ln_1")
        self.mlp_dense_1 = keras.layers.Dense(
            self.proj_dim * 4,
            name="c_fc",
        )
        self.mlp_activation = QuickGELU(name="gelu")
        self.mlp_dense_2 = keras.layers.Dense(
            self.proj_dim,
            name="c_proj",
        )
        self.ln_2 = keras.layers.LayerNormalization(epsilon=1e-5, name="ln_2")

    def attention(self, x, causal_attention_mask=None, attention_mask=None):
        mask = None
        if causal_attention_mask is not None:
            mask = (
                ops.cast(causal_attention_mask, dtype=x.dtype)
                if causal_attention_mask is not None
                else None
            )
        if attention_mask is not None:
            attention_mask = (
                ops.cast(attention_mask, dtype=x.dtype)
                if attention_mask is not None
                else None
            )
            mask = ops.add(causal_attention_mask, attention_mask)

        return self.attn(
            x,
            attention_mask=mask,
        )[0]

    def build(self, input_shape):
        super().build(input_shape)
        self.attn.build(None)
        self.ln_1.build([None, None, self.proj_dim])
        self.mlp_dense_1.build([None, None, self.proj_dim])
        self.mlp_dense_2.build([None, None, self.proj_dim * 4])
        self.ln_2.build([None, None, self.proj_dim])

    def call(self, x, causal_attention_mask=None, attention_mask=None):
        residual = x
        x = self.ln_1(x)
        x = self.attention(
            x,
            causal_attention_mask=causal_attention_mask,
            attention_mask=attention_mask,
        )
        x = x + residual
        residual = x
        x = self.mlp_dense_1(self.ln_2(residual))
        x = self.mlp_activation(x)
        x = self.mlp_dense_2(x)
        x = residual + x
        return x

    def compute_output_shape(self, inputs_shape):
        return inputs_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "proj_dim": self.proj_dim,
                "num_heads": self.num_heads,
                "num_hidden_layers": self.num_hidden_layers,
            }
        )
        return config


@keras_cv_export("keras_cv.models.feature_extractor.CLIPEncoder")
class CLIPEncoder(keras.layers.Layer):
    def __init__(self, width, num_layers, heads, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.num_layers = num_layers
        self.heads = heads
        self.resblocks = [
            ResidualAttention(
                self.width,
                self.heads,
                self.num_layers,
            )
            for _ in range(self.num_layers)
        ]

    def build(self, input_shape):
        for block in self.resblocks:
            block.build(input_shape)
        self.built = True

    def compute_output_shape(self, input_shape):
        for block in self.resblocks:
            input_shape = block.compute_output_shape(input_shape)
        return input_shape

    def call(
        self,
        x,
        causal_attention_mask=None,
        attention_mask=None,
    ):
        for block in self.resblocks:
            x = block(
                x,
                causal_attention_mask=causal_attention_mask,
                attention_mask=attention_mask,
            )
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "width": self.width,
                "num_layers": self.num_layers,
                "heads": self.heads,
            }
        )
        return config


@keras_cv_export("keras_cv.models.feature_extractor.CLIPAttention")
class CLIPAttention(keras.layers.Layer):
    """
    Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py # noqa: E501
    """

    def __init__(
        self, proj_dim, num_heads, num_hidden_layers, dropout=0.0, **kwargs
    ):
        super().__init__(**kwargs)

        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.head_dim = self.proj_dim // self.num_heads
        if self.head_dim * self.num_heads != self.proj_dim:
            raise ValueError(
                f"proj_dim must be divisible by num_heads (got `proj_dim`"
                f": {self.proj_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale = self.head_dim**-0.5
        self.q_proj = keras.layers.Dense(
            units=self.proj_dim,
            name="q_proj",
        )
        self.k_proj = keras.layers.Dense(
            units=self.proj_dim,
            name="k_proj",
        )
        self.v_proj = keras.layers.Dense(
            units=self.proj_dim,
            name="v_proj",
        )
        self.out_proj = keras.layers.Dense(
            units=self.proj_dim,
            name="out_proj",
        )

    def build(self, input_shape):
        super().build(input_shape)
        self.q_proj.build([None, None, self.proj_dim])
        self.k_proj.build([None, None, self.proj_dim])
        self.v_proj.build([None, None, self.proj_dim])
        self.out_proj.build([None, None, self.proj_dim])

    def _transpose_for_scores(self, tensor, batch_size):
        """
        Adapted from https://github.com/huggingface/transformers/blob/8e164c5400b7b413c7b8fb32e35132001effc970/src/transformers/models/bert/modeling_tf_bert.py#L252 # noqa: E501
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

        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in the
            # call() function)
            attention_scores = ops.add(attention_scores, attention_mask)

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        dropout_attention_probs = keras.layers.Dropout(self.dropout)(
            inputs=attention_probs, training=training
        )

        attn_output = ops.matmul(dropout_attention_probs, value_layer)
        attn_output = ops.transpose(attn_output, axes=[0, 2, 1, 3])

        # (batch_size, seq_len_q, proj_dim)
        attn_output = ops.reshape(attn_output, (batch_size, -1, self.proj_dim))

        attn_output = self.out_proj(attn_output, training=training)
        outputs = (
            (attn_output, attention_probs)
            if output_attentions
            else (attn_output,)
        )

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "proj_dim": self.proj_dim,
                "num_heads": self.num_heads,
                "num_hidden_layers": self.num_hidden_layers,
                "dropout": self.dropout,
            }
        )
        return config
