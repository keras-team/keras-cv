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

import keras_cv.layers.maxvit_layers as maxvit_layers


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
        activation: default 'tf.activations.gelu', the activation function to apply in the MLP head - should be a function
        layer_norm_epsilon: default 1e-06, the epsilon for `LayerNormalization` layers

    Basic usage:

    ```
    project_dim = 1024
    mlp_dim = 3072
    num_heads = 4

    encoded_patches = keras_cv.layers.PatchingAndEmbedding(project_dim=project_dim, patch_size=16)(img_batch)
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
        activation=tf.keras.activations.gelu,
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
        self.dense1 = layers.Dense(self.mlp_units[0])
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

        x = self.layer_norm1(inputs)
        x = self.attn(x, x)
        x = layers.Dropout(self.mlp_dropout)(x)
        x = layers.Add()([x, inputs])

        y = self.layer_norm2(x)

        y = self.dense1(y)
        if self.activation == tf.keras.activations.gelu:
            y = self.activation(y, approximate=True)
        else:
            y = self.activation(y)
        y = layers.Dropout(self.mlp_dropout)(y)
        y = self.dense2(y)
        y = layers.Dropout(self.mlp_dropout)(y)

        output = layers.Add()([x, y])

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

    @classmethod
    def from_config(cls, config, custom_objects=None):
        activation = config.pop("activation")
        activation = tf.keras.activations.deserialize(activation)
        return cls(activation=activation, **config)


# Never actually used in MaxViTs but could conceivably be used
# to generalize: LN -> grid/window -> Attn -> residual add -> LN -> FFN
@tf.keras.utils.register_keras_serializable(package="keras_cv")
class MaxViTTransformerEncoder(layers.Layer):
    # Attention + FFN (LN + Attention + Residual + LN + MLP)
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
        dropatt=None,
        rel_attn_type="2d_multi_head",
        scale_ratio=None,
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
        self.dropatt = dropatt
        self.rel_attn_type = rel_attn_type
        self.scale_ratio = scale_ratio
        self.ln_epsilon = ln_epsilon
        self.ln_dtype = ln_dtype
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        # BlockAttention Layer norm
        self.block_attn_layer_norm = tf.keras.layers.LayerNormalization(
            axis=-1,
            epsilon=self.ln_epsilon,
            dtype=self.ln_dtype,
            name="attn_layer_norm",
        )
        self.window_partition = maxvit_layers.WindowPartitioning(
            window_size=self.window_size
        )
        # Unblock
        self.unwindow_partition = maxvit_layers.UnWindowPartitioning(
            window_size=self.window_size
        )

        # Relative Attention
        self.hidden_size = hidden_size
        if self.num_heads is None:
            self.num_heads = self.hidden_size // self.head_size

        self.block_attention = maxvit_layers.RelativeMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.head_size,
            value_dim=self.head_size,
            dropout=self.dropatt,
            use_bias=True,
        )

        self.grid_attn_layer_norm = layers.LayerNormalization(
            axis=-1,
            epsilon=self.ln_epsilon,
            dtype=self.ln_dtype,
            name="attn_layer_norm_1",
        )

        self.grid_partition = maxvit_layers.GridPartitioning(self.grid_size)
        self.ungrid_partition = maxvit_layers.UnGridPartitioning(
            grid_size=self.grid_size
        )

        # Relative Attention

        self.grid_attention = maxvit_layers.RelativeMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.head_size,
            value_dim=self.head_size,
            dropout=self.dropatt,
            use_bias=True,
        )

        self.block_ffn_layer_norm = layers.LayerNormalization(
            axis=-1,
            epsilon=self.ln_epsilon,
            dtype=self.ln_dtype,
            name="block_ffn_layer_norm_1",
        )
        self.block_ffn = maxvit_layers._FFN(
            self.hidden_size,
            bias_initializer=self.bias_initializer,
            kernel_initializer=self.kernel_initializer,
        )

        self.grid_ffn_layer_norm = layers.LayerNormalization(
            axis=-1,
            epsilon=self.ln_epsilon,
            dtype=self.ln_dtype,
            name="grid_ffn_layer_norm_1",
        )
        self.grid_ffn = maxvit_layers._FFN(
            self.hidden_size,
            bias_initializer=self.bias_initializer,
            kernel_initializer=self.kernel_initializer,
        )

    def call(self, input):
        x = self.block_attn_layer_norm(input)
        shortcut = x

        # For unwindowing
        _, height, width, _ = x.shape

        x = self.window_partition(x)
        x = self.__reshape_to_1d(x)
        x = self.block_attention(x, x)
        x = self.unwindow_partition(x, height, width)
        if self.dropout:
            x = layers.Dropout(self.dropout)(x)
        x = layers.Add()([shortcut, x])

        # Grid-Attention
        x = self.grid_attn_layer_norm(x)
        shortcut = x
        # For ungridding
        _, height, width, _ = x.shape
        x = self.grid_partition(x)
        x = self.__reshape_to_1d(x)
        x = self.grid_attention(x, x)
        x = self.ungrid_partition(x, height, width)
        if self.dropout:
            x = layers.Dropout(self.dropout)(x)
        x = layers.Add()([shortcut, x])
        return x

    """
    Taken from: https://github.com/google-research/maxvit/blob/2e06a7f1f70c76e64cd3dabe5cd1b8c1a23c9fb7/maxvit/models/common_ops.py#L129
    """

    def __reshape_to_1d(self, x):
        """Reshape tensor to 1d if not already 1d."""
        if x.shape.rank == 4:
            _, h, w, num_channel = x.shape.as_list()
            return tf.reshape(x, [-1, h * w, num_channel])
        elif x.shape.rank == 3:
            return x
        else:
            raise ValueError("Unsupported shape {}".format(x.shape))
