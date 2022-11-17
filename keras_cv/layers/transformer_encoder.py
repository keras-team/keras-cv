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
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class TransformerEncoder(layers.Layer):
    """
    Transformer encoder block implementation as a Keras Layer.

    Args:
        project_dim: the dimensionality of the projection of the encoder, and output of the `MultiHeadAttention`
        mlp_dim: the intermediate dimensionality of the MLP head before projecting to `project_dim`
        num_heads: the number of heads for the `MultiHeadAttention` layer
        mlp_dropout: default 0.1, the dropout rate to apply between the layers of the MLP head of the encoder
        attention_dropout: default 0.1, the dropout rate to apply in the MultiHeadAttention layer
        activation: default 'gelu', the activation function to apply in the MLP head
        layer_norm_epsilon: default 1e-06, the epsilon for `LayerNormalization` layers

    Basic usage:

    ```
    project_dim = 1024
    mlp_dim = 3072
    num_heads = 4

    patches = keras_cv.layers.Patching(patch_size)(cat_img)
    encoded_patches = keras_cv.layers.PatchEmbedding(project_dim=project_dim)(patches)
    trans_encoded = keras_cv.layers.TransformerEncoder(project_dim=project_dim,
                                                       mlp_dim = mlp_dim,
                                                       num_heads=num_heads)(encoded_patches)

    print(trans_encoded.shape) # (1, 197, 1024)
    ```
    """

    def __init__(
        self,
        project_dim,
        num_heads,
        mlp_dim,
        mlp_dropout=0.1,
        attention_dropout=0.1,
        activation="gelu",
        layer_norm_epsilon=1e-06,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.mlp_dropout = mlp_dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.mlp_units = [mlp_dim, project_dim]

        self.layer_norm1 = layers.LayerNormalization(epsilon=self.layer_norm_epsilon)
        self.layer_norm2 = layers.LayerNormalization(epsilon=self.layer_norm_epsilon)
        self.attn = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.project_dim // self.num_heads,
            dropout=self.attention_dropout,
        )
        self.dense1 = layers.Dense(self.mlp_units[0], activation=activation)
        self.dense2 = layers.Dense(self.mlp_units[1])

    def call(self, inputs):
        """Calls the Transformer Encoder on an input sequence.
        Args:
            inputs: A `tf.Tensor` of shape [batch, height, width, channels]

        Returns:
            `A tf.Tensor` of shape [batch, patch_num+1, embedding_dim]
        """

        if inputs.shape[-1] != self.project_dim:
            raise ValueError(
                f"The input and output dimensionality must be the same, but the TransformerEncoder was provided with {inputs.shape[-1]} and {self.project_dim}"
            )

        x1 = self.layer_norm1(inputs)
        attention_output = self.attn(x1, x1)
        x2 = layers.Add()([attention_output, inputs])
        x3 = self.layer_norm2(x2)
        x3 = self.dense1(x3)
        x3 = layers.Dropout(self.mlp_dropout)(x3)
        x3 = self.dense2(x3)
        x3 = layers.Dropout(self.mlp_dropout)(x3)

        output = layers.Add()([x3, x2])

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "project_dim": self.project_dim,
                "mlp_dim": self.mlp_dim,
                "num_heads": self.num_heads,
                "attention_dropout": self.attention_dropout,
                "mlp_dropout": self.mlp_dropout,
                "activation": self.activation,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
