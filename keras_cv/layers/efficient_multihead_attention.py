from keras_cv.backend import keras
from keras_cv.backend import ops


@keras.saving.register_keras_serializable(package="keras_cv")
class EfficientMultiheadAttention(keras.layers.Layer):
    def __init__(self, project_dim, num_heads, sr_ratio):
        """
        Efficient MultiHeadAttention implementation as a Keras layer.
        A huge bottleneck in scaling transformers is the self-attention layer with an O(n^2) complexity.

        EfficientMultiHeadAttention performs a sequence reduction (SR) operation with a given ratio, to reduce
        the sequence length before performing key and value projections, reducing the O(n^2) complexity to O(n^2/R) where
        R is the sequence reduction ratio.

        References:
        - [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) (CVPR 2021)
        - [NVlabs' official implementation](https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/backbones/mix_transformer.py)
        - [@sithu31296's reimplementation](https://github.com/sithu31296/semantic-segmentation/blob/main/semseg/models/backbones/mit.py)
        - [Ported from the TensorFlow implementation from DeepVision](https://github.com/DavidLandup0/deepvision/blob/main/deepvision/layers/efficient_attention.py)

        Args:
            project_dim: the dimensionality of the projection of the `EfficientMultiHeadAttention` layer.
            num_heads: the number of heads to use in the attention computation.
            sr_ratio: the sequence reduction ratio to perform on the sequence before key and value projections.

        Basic usage:

        ```
        tensor = tf.random.uniform([1, 196, 32])
        output = keras_cv.layers.EfficientMultiheadAttention(project_dim=768,
                                                            num_heads=2,
                                                            sr_ratio=4)(tensor)
        print(output.shape) # (1, 196, 32)
        ```
        """
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
        input_shape = ops.shape(x)
        H, W = ops.sqrt(ops.cast(input_shape[1], "float32")), ops.sqrt(
            ops.cast(input_shape[1], "float32")
        )
        B, C = ops.cast(input_shape[0], "float32"), ops.cast(
            input_shape[2], "float32"
        )

        q = self.q(x)

        q = ops.reshape(
            q,
            (
                input_shape[0],
                input_shape[1],
                self.num_heads,
                input_shape[2] // self.num_heads,
            ),
        )
        q = ops.transpose(q, [0, 2, 1, 3])

        if self.sr_ratio > 1:
            x = ops.reshape(
                ops.transpose(x, [0, 2, 1]),
                (B, H, W, C),
            )
            x = self.sr(x)
            x = ops.reshape(x, [input_shape[0], input_shape[2], -1])
            x = ops.transpose(x, [0, 2, 1])
            x = self.norm(x)

        k = self.k(x)
        v = self.v(x)

        k = ops.transpose(
            ops.reshape(
                k,
                [B, -1, self.num_heads, C // self.num_heads],
            ),
            [0, 2, 1, 3],
        )

        v = ops.transpose(
            ops.reshape(
                v,
                [B, -1, self.num_heads, C // self.num_heads],
            ),
            [0, 2, 1, 3],
        )

        attn = (q @ ops.transpose(k, [0, 1, 3, 2])) * self.scale
        attn = ops.nn.softmax(attn, axis=-1)

        attn = attn @ v
        attn = ops.reshape(
            ops.transpose(attn, [0, 2, 1, 3]),
            [input_shape[0], input_shape[1], input_shape[2]],
        )
        x = self.proj(attn)
        return x
