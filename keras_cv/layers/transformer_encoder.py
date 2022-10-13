"""Transformer encoder block implementation as a Keras Layer."""

from tensorflow import keras
from tensorflow.keras import layers
from keras_cv.transformers import mlp_ffn

@tf.keras.utils.register_keras_serializable(package="keras_cv")
class TransformerEncoder(layers.Layer):
    def __init__(self, project_dims,
                 num_heads,
                 dropout=0.1,
                 activation="relu",
                 layer_norm_epsilon=1e-05,
                 transformer_units=None):
        super(TransformerEncoder, self).__init__()

        self.project_dim = project_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.transformer_units = transformer_units

    def call(self, input):
        x1 = layers.LayerNormalization(epsilon=1e-6)(input)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.project_dim, dropout=self.dropout
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, input])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        # Hardcode for now
        transformer_units = [
            self.project_dim * 2,
            self.project_dim,
        ]
        x3 = mlp_ffn.mlp_ffn(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
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
