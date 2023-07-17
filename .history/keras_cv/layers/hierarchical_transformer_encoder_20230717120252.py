from keras_cv.backend import keras
from keras_cv.layers import StochasticDepth


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
            project_dim, num_heads, sr_ratio, backend="tensorflow"
        )
        self.drop_path = StochasticDepth(drop_prob, backend="tensorflow")
        self.norm2 = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)
        self.mlp = MixFFN(
            channels=project_dim,
            mid_channels=int(project_dim * 4),
            backend="tensorflow",
        )

    def call(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x
