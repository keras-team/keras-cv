from typing import Tuple

import tensorflow as tf
from segformer_utils import shape_list


class OverlapPatchEmbeddings(tf.keras.layers.Layer):
    """Construct the overlapping patch embeddings."""

    def __init__(self, patch_size, stride, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.padding = tf.keras.layers.ZeroPadding2D(padding=patch_size // 2)
        self.proj = tf.keras.layers.Conv2D(
            filters=hidden_size,
            kernel_size=patch_size,
            strides=stride,
            padding="VALID",
            name="proj",
        )

        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-05, name="layer_norm"
        )

    def call(self, pixel_values: tf.Tensor) -> Tuple[tf.Tensor, int, int]:
        embeddings = self.proj(self.padding(pixel_values))
        height = shape_list(embeddings)[1]
        width = shape_list(embeddings)[2]
        hidden_dim = shape_list(embeddings)[3]
        # (batch_size, height, width, num_channels) -> (batch_size, height*width, num_channels)
        # this can be fed to a Transformer layer
        embeddings = tf.reshape(embeddings, (-1, height * width, hidden_dim))
        embeddings = self.layer_norm(embeddings)
        return embeddings, height, width
