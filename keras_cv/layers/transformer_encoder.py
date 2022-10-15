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
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable(package="keras_cv")
class TransformerEncoder(layers.Layer):
    """
    Transformer encoder block implementation as a Keras Layer.
    args:
        - project_dim: the dimensionality of the projection of the encoder
        - num_heads: the number of heads for the `MultiHeadAttention` layer
        - dropout: default 0.1, the dropout rate to apply inside the MLP head of the encoder
        - activation: default tf.nn.gelu(), the activation function to apply in the MLP head
        - layer_norm_epsilon: default 1e-06, the epsilon for `LayerNormalization` layers
        - transformer_units: list of dimensionalities for the `Dense` layers in the MLP head
    """
    def __init__(self, project_dim,
                 num_heads,
                 dropout=0.1,
                 activation=tf.nn.gelu,
                 layer_norm_epsilon=1e-06,
                 transformer_units=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.project_dim = project_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.transformer_units = transformer_units

    def call(self, input):
        x1 = layers.LayerNormalization(epsilon=self.layer_norm_epsilon)(input)
        attention_output = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.project_dim, dropout=self.dropout
        )(x1, x1)
        x2 = layers.Add()([attention_output, input])
        x3 = layers.LayerNormalization(epsilon=self.layer_norm_epsilon)(x2)
        x3 = mlp_head(x3, dropout_rate=self.dropout, hidden_units=self.transformer_units)
        encoded_patches = layers.Add()([x3, x2])

        return encoded_patches

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "project_dim": self.project_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "transformer_units": self.transformer_units
            }
        )
        return config

"""
Helper method for creating an MLP head
"""
def mlp_head(x, dropout_rate, hidden_units, activation):
    for (idx, units) in enumerate(hidden_units):
        x = layers.Dense(units, activation=activation if idx == 0 else None)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x