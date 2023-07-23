import math

from keras_cv.backend import keras
from keras_cv.layers.efficient_multihead_attention import (
    EfficientMultiheadAttention,
)
from keras_cv.layers.regularization.drop_path import DropPath


@keras.saving.register_keras_serializable(package="keras_cv")
class HierarchicalTransformerEncoder(keras.layers.Layer):
    def __init__(
        self,
        project_dim,
        num_heads,
        sr_ratio=1,
        drop_prob=0.0,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm1 = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)
        self.attn = EfficientMultiheadAttention(
            project_dim, num_heads, sr_ratio
        )
        self.drop_path = DropPath(drop_prob)
        self.norm2 = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)
        self.mlp = self.MixFFN(
            channels=project_dim,
            mid_channels=int(project_dim * 4),
        )

    def call(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    class MixFFN(keras.layers.Layer):
        def __init__(self, channels, mid_channels):
            super().__init__()
            self.fc1 = keras.layers.Dense(mid_channels)
            self.dwconv = keras.layers.DepthwiseConv2D(
                kernel_size=3,
                strides=1,
                padding="same",
            )
            self.fc2 = keras.layers.Dense(channels)

        def call(self, x):
            x = self.fc1(x)
            shape = keras.ops.shape(x)
            B, C = keras.ops.cast(shape[0], "float32"), keras.ops.cast(
                shape[-1], "float32"
            )
            H, W = keras.ops.sqrt(
                keras.ops.cast(shape[1], "float32")
            ), keras.ops.sqrt(keras.ops.cast(shape[1], "float32"))
            # print(B, C, H, W)
            x = keras.ops.reshape(x, (B, H, W, C))
            x = self.dwconv(x)
            x = keras.ops.reshape(x, (B, -1, C))
            x = keras.ops.nn.gelu(x)
            x = self.fc2(x)
            return x
