from keras_cv.backend import keras
from keras_cv.layers.efficient_multihead_attention import (
    EfficientMultiheadAttention,
)
from keras_cv.layers.regularization.stochastic_depth import StochasticDepth


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
        self.drop_path = StochasticDepth(drop_prob)
        self.norm2 = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)
        self.mlp = self.__MixFFN(
            channels=project_dim,
            mid_channels=int(project_dim * 4),
        )

    def call(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

    class __MixFFN(keras.layers.Layer):
        def __init__(self, channels, mid_channels):
            super().__init__()
            self.fc1 = keras.layers.Dense(mid_channels)
            self.dwconv = keras.layers.DepthwiseConv2D(
                kernel_size=3,
                strides=1,
                padding="same",
            )
            self.fc2 = keras.layers.Dense(channels)

        def call(self, x, H, W):
            x = self.fc1(x)
            # B, DIM, C
            input_shape = x.shape

            x = keras.ops.reshape(x, (input_shape[0], H, W, input_shape[-1]))
            x = self.dwconv(x)
            x = keras.ops.reshape(x, (input_shape[0], -1, input_shape[-1]))
            x = keras.nn.ops.gelu(x)
            x = self.fc2(x)
            return x
