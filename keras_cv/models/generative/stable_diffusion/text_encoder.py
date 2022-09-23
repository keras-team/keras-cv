import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.experimental import numpy as tfnp


class TextEncoder(keras.Model):
    def __init__(self, max_length, name=None):
        input_tokens = keras.layers.Input(
            shape=(max_length,), dtype="int32", name="input_tokens"
        )
        token_positions = keras.layers.Input(
            shape=(max_length,), dtype="int32", name="token_positions"
        )

        input_dim, output_dim = 49408, 768

        x = CLIPEmbedding(input_dim, output_dim, max_length)(
            [input_tokens, token_positions]
        )
        for _ in range(12):
            x = CLIPEncoderLayer()(x)
        embedded = keras.layers.LayerNormalization(epsilon=1e-5)(x)
        super().__init__([input_tokens, token_positions], embedded, name=name)


class CLIPAttention(keras.layers.Layer):
    def __init__(self, embed_dim=768, num_heads=12, causal=True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = keras.layers.Dense(self.embed_dim)
        self.k_proj = keras.layers.Dense(self.embed_dim)
        self.v_proj = keras.layers.Dense(self.embed_dim)
        self.out_proj = keras.layers.Dense(self.embed_dim)

    def _shape(self, tensor, sequence_length, batch_size):
        a = tf.reshape(
            tensor, (batch_size, sequence_length, self.num_heads, self.head_dim)
        )
        return tf.transpose(a, (0, 2, 1, 3))  # bs , n_head , sequence_length , head_dim

    def call(self, inputs, attention_mask=None):
        if attention_mask is None and self.causal:
            length = tf.shape(inputs)[1]
            attention_mask = tfnp.triu(tf.ones((1, 1, length, length)) * -tfnp.inf, k=1)

        batch_size, tgt_len, embed_dim = inputs.shape
        query_states = self.q_proj(inputs) * self.scale
        key_states = self._shape(self.k_proj(inputs), tgt_len, -1)
        value_states = self._shape(self.v_proj(inputs), tgt_len, -1)

        proj_shape = (-1, tgt_len, self.head_dim)
        query_states = self._shape(query_states, tgt_len, -1)
        query_states = tf.reshape(query_states, proj_shape)
        key_states = tf.reshape(key_states, proj_shape)

        src_len = tgt_len
        value_states = tf.reshape(value_states, proj_shape)
        attn_weights = query_states @ tf.transpose(key_states, (0, 2, 1))

        attn_weights = tf.reshape(attn_weights, (-1, self.num_heads, tgt_len, src_len))
        attn_weights = attn_weights + attention_mask
        attn_weights = tf.reshape(attn_weights, (-1, tgt_len, src_len))

        attn_weights = tf.nn.softmax(attn_weights)
        attn_output = attn_weights @ value_states

        attn_output = tf.reshape(
            attn_output, (-1, self.num_heads, tgt_len, self.head_dim)
        )
        attn_output = tf.transpose(attn_output, (0, 2, 1, 3))
        attn_output = tf.reshape(attn_output, (-1, tgt_len, embed_dim))

        return self.out_proj(attn_output)


class CLIPEncoderLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.self_attn = CLIPAttention(causal=True)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.fc1 = keras.layers.Dense(3072)
        self.fc2 = keras.layers.Dense(768)

    def call(self, inputs):
        residual = inputs

        hidden_states = self.layer_norm1(inputs)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = quick_gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)

        return residual + hidden_states


class CLIPEmbedding(keras.layers.Layer):
    def __init__(self, input_dim=49408, output_dim=768, max_length=77, **kwargs):
        super().__init__(**kwargs)
        self.token_embedding_layer = keras.layers.Embedding(
            input_dim, output_dim, name="token_embedding"
        )
        self.position_embedding_layer = keras.layers.Embedding(
            max_length, output_dim, name="position_embedding"
        )

    def call(self, inputs):
        input_ids, position_ids = inputs
        word_embeddings = self.token_embedding_layer(input_ids)
        position_embeddings = self.position_embedding_layer(position_ids)
        return word_embeddings + position_embeddings


def quick_gelu(x):
    return x * tf.sigmoid(x * 1.702)
