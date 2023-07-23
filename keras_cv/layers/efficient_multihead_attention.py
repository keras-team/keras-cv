import math

from keras_cv.backend import keras

"""
Based on: https://github.com/sithu31296/semantic-segmentation/blob/main/semseg/models/backbones/mit.py
"""


@keras.saving.register_keras_serializable(package="keras_cv")
class EfficientMultiheadAttention(keras.layers.Layer):
    def __init__(self, project_dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.scale = (project_dim // num_heads) ** -0.5
        self.q = keras.layers.Dense(project_dim)
        self.k = keras.layers.Dense(project_dim)
        self.v = keras.layers.Dense(project_dim)
        self.proj = keras.layers.Dense(project_dim)

        if sr_ratio > 1:
            self.sr = keras.layers.Conv2D(
                filters=project_dim,
                kernel_size=sr_ratio,
                strides=sr_ratio,
                padding="same",
            )
            self.norm = keras.layers.LayerNormalization()

    def call(self, x):
        input_shape = keras.ops.shape(x)
        H, W = keras.ops.sqrt(
            keras.ops.cast(input_shape[1], "float32")
        ), keras.ops.sqrt(keras.ops.cast(input_shape[1], "float32"))
        B, C = keras.ops.cast(input_shape[0], "float32"), keras.ops.cast(
            input_shape[2], "float32"
        )

        q = self.q(x)

        q = keras.ops.reshape(
            q,
            (
                input_shape[0],
                input_shape[1],
                self.num_heads,
                input_shape[2] // self.num_heads,
            ),
        )
        q = keras.ops.transpose(q, [0, 2, 1, 3])

        if self.sr_ratio > 1:
            x = keras.ops.reshape(
                keras.ops.transpose(x, [0, 2, 1]),
                (B, H, W, C),
            )
            x = self.sr(x)
            x = keras.ops.reshape(x, [input_shape[0], input_shape[2], -1])
            x = keras.ops.transpose(x, [0, 2, 1])
            x = self.norm(x)

        k = self.k(x)
        v = self.v(x)

        k = keras.ops.transpose(
            keras.ops.reshape(
                k,
                [B, -1, self.num_heads, C // self.num_heads],
            ),
            [0, 2, 1, 3],
        )

        v = keras.ops.transpose(
            keras.ops.reshape(
                v,
                [B, -1, self.num_heads, C // self.num_heads],
            ),
            [0, 2, 1, 3],
        )

        attn = (q @ keras.ops.transpose(k, [0, 1, 3, 2])) * self.scale
        attn = keras.ops.nn.softmax(attn, axis=-1)

        attn = attn @ v
        attn = keras.ops.reshape(
            keras.ops.transpose(attn, [0, 2, 1, 3]),
            [input_shape[0], input_shape[1], input_shape[2]],
        )
        x = self.proj(attn)
        return x
