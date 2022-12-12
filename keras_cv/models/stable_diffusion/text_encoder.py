# Copyright 2022 The KerasCV Authors
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

import tensorflow as tf
from tensorflow import keras
from tensorflow.experimental import numpy as tfnp


class TextEncoder(keras.Model):
    def __init__(self, config, name=None, download_weights=True):
        tokens = keras.layers.Input(shape=[config['max_length'],], dtype="int32", name="tokens")
        positions = keras.layers.Input(
            shape=[config['max_length'],], dtype="int32", name="positions"
        )
        x = CLIPEmbedding(config['vocab_size'], config['embed_dim'], config['max_length'])([tokens, positions])
        for _ in range(config['num_blocks']):
            x = CLIPEncoderLayer(
                embed_dim=config['embed_dim'], 
                num_heads=config['num_heads'], 
                use_q_gelu=config['use_q_gelu']
                )(x)
        embedded = keras.layers.LayerNormalization(epsilon=1e-5)(x)
        super().__init__([tokens, positions], embedded, name=name)

        if download_weights:
            if config['version']=='v1':
                text_encoder_weights_fpath = keras.utils.get_file(
                    origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_encoder.h5",
                    file_hash="4789e63e07c0e54d6a34a29b45ce81ece27060c499a709d556c7755b42bb0dc4",
                )
            else:
                text_encoder_weights_fpath = keras.utils.get_file(
                    origin="https://huggingface.co/datasets/Jobayer/StableDiffusion/resolve/main/text_encoder.h5",
                    file_hash="9ad00b5d2f10d297d122ddf4c0cacc643a6e36a1576b43060239f35ecb5e965a",
                )
            self.load_weights(text_encoder_weights_fpath)


class CLIPEmbedding(keras.layers.Layer):
    def __init__(self, input_dim=49408, embed_dim=768, max_length=77, **kwargs):
        super().__init__(**kwargs)
        self.token_embedding = keras.layers.Embedding(input_dim, embed_dim)
        self.position_embedding = keras.layers.Embedding(max_length, embed_dim)

    def call(self, inputs):
        tokens, positions = inputs
        tokens = self.token_embedding(tokens)
        positions = self.position_embedding(positions)
        return tokens + positions

def quick_gelu(x):
    return x * tf.sigmoid(x * 1.702)

class CLIPEncoderLayer(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, use_q_gelu=True, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.clip_attn = CLIPAttention(embed_dim=embed_dim, num_heads=num_heads, causal=True)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.fc1 = keras.layers.Dense(embed_dim*4)
        self.fc2 = keras.layers.Dense(embed_dim)
        self.gelu = quick_gelu if use_q_gelu else tf.nn.gelu
    def call(self, inputs):
        residual = inputs
        x = self.layer_norm1(inputs)
        x = self.clip_attn(x)
        x = residual + x
        residual = x
        x = self.layer_norm2(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x + residual


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

    def reshape_states(self, x, sequence_length, batch_size):
        x = tf.reshape(x, (batch_size, sequence_length, self.num_heads, self.head_dim))
        return tf.transpose(x, (0, 2, 1, 3))  # bs, heads, sequence_length, head_dim

    def call(self, inputs, attention_mask=None):
        if attention_mask is None and self.causal:
            length = tf.shape(inputs)[1]
            attention_mask = tfnp.triu(
                tf.ones((1, 1, length, length), dtype=self.compute_dtype) * -tfnp.inf,
                k=1,
            )

        _, tgt_len, embed_dim = inputs.shape
        query_states = self.q_proj(inputs) * self.scale
        key_states = self.reshape_states(self.k_proj(inputs), tgt_len, -1)
        value_states = self.reshape_states(self.v_proj(inputs), tgt_len, -1)

        proj_shape = (-1, tgt_len, self.head_dim)
        query_states = self.reshape_states(query_states, tgt_len, -1)
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
