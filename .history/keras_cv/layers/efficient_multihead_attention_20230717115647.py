from keras_cv.backend import keras

"""
Based on: https://github.com/sithu31296/semantic-segmentation/blob/main/semseg/models/backbones/mit.py
"""


@keras.saving.register_keras_serializable(package="keras_cv")
class __EfficientMultiheadAttentionPT(nn.Module):
    def __init__(self, project_dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.scale = (project_dim // num_heads) ** -0.5
        self.q = nn.Linear(project_dim, project_dim)
        self.kv = nn.Linear(project_dim, project_dim * 2)
        self.proj = nn.Linear(project_dim, project_dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                in_channels=project_dim,
                out_channels=project_dim,
                kernel_size=sr_ratio,
                stride=sr_ratio,
                padding=same_padding(sr_ratio, sr_ratio),
            )
            self.norm = nn.LayerNorm(project_dim)

    def forward(self, x, H, W):
        batch_size, seq_len, project_dim = x.shape
        q = (
            self.q(x)
            .reshape(
                batch_size,
                seq_len,
                self.num_heads,
                project_dim // self.num_heads,
            )
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(batch_size, project_dim, H, W)
            x = self.sr(x).reshape(batch_size, project_dim, -1).permute(0, 2, 1)
            x = self.norm(x)

        k, v = (
            self.kv(x)
            .reshape(
                batch_size, -1, 2, self.num_heads, project_dim // self.num_heads
            )
            .permute(2, 0, 3, 1, 4)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, project_dim)
        x = self.proj(x)
        return x


class __EfficientMultiheadAttentionTF(tf.keras.layers.Layer):
    def __init__(self, project_dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.scale = (project_dim // num_heads) ** -0.5
        self.q = tf.keras.layers.Dense(project_dim)
        self.k = tf.keras.layers.Dense(project_dim)
        self.v = tf.keras.layers.Dense(project_dim)
        self.proj = tf.keras.layers.Dense(project_dim)

        if sr_ratio > 1:
            self.sr = tf.keras.layers.Conv2D(
                filters=project_dim,
                kernel_size=sr_ratio,
                strides=sr_ratio,
                padding="same",
            )
            self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x, H, W):
        input_shape = tf.shape(x)

        q = self.q(x)
        q = tf.reshape(
            q,
            shape=[
                input_shape[0],
                input_shape[1],
                self.num_heads,
                input_shape[2] // self.num_heads,
            ],
        )

        q = tf.transpose(q, [0, 2, 1, 3])

        if self.sr_ratio > 1:
            x = tf.reshape(
                tf.transpose(x, [0, 2, 1]),
                shape=[input_shape[0], H, W, input_shape[2]],
            )
            x = self.sr(x)
            x = tf.reshape(x, [input_shape[0], input_shape[2], -1])
            x = tf.transpose(x, [0, 2, 1])
            x = self.norm(x)

        k = self.k(x)
        v = self.v(x)

        k = tf.transpose(
            tf.reshape(
                k,
                [
                    input_shape[0],
                    -1,
                    self.num_heads,
                    input_shape[2] // self.num_heads,
                ],
            ),
            [0, 2, 1, 3],
        )

        v = tf.transpose(
            tf.reshape(
                v,
                [
                    input_shape[0],
                    -1,
                    self.num_heads,
                    input_shape[2] // self.num_heads,
                ],
            ),
            [0, 2, 1, 3],
        )

        attn = (q @ tf.transpose(k, [0, 1, 3, 2])) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)

        attn = attn @ v
        attn = tf.transpose(attn, [0, 2, 1, 3])
        attn = tf.reshape(
            attn, shape=[input_shape[0], input_shape[1], input_shape[2]]
        )
        x = self.proj(attn)
        return x


LAYER_BACKBONES = {
    "tensorflow": __EfficientMultiheadAttentionTF,
    "pytorch": __EfficientMultiheadAttentionPT,
}


def EfficientMultiheadAttention(
    project_dim, num_heads, sr_ratio, backend="pytorch"
):
    """
    `EfficientMultiheadAttention` is a standard scaled softmax attention layer, but shortens the sequence it operates on by a reduction factor, to reduce computational cost.
    The layer is meant to be used as part of the `deepvision.layers.HierarchicalTransformerEncoder` for the SegFormer architecture.

    Reference:
        - ["SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"](https://arxiv.org/pdf/2105.15203v2.pdf)

    Args:
        project_dim: the dimensionality of the projection for the keys, values and queries
        num_heads: the number of attention heads to apply
        sr_ratio: the reduction ratio for the sequence length
        backend: the backend framework to use

    Basic usage:

    ```
    tensor = torch.rand(1, 196, 32)
    output = deepvision.layers.EfficientMultiheadAttention(project_dim=32,
                                                  num_heads=2,
                                                  sr_ratio=4,
                                                  backend='pytorch')(tensor, H=14, W=14)

    print(output.shape) # torch.Size([1, 196, 32])

    tensor = tf.random.uniform([1, 196, 32])
    output = deepvision.layers.EfficientMultiheadAttention(project_dim=32,
                                                  num_heads=2,
                                                  sr_ratio=4,
                                                  backend='tensorflow')(tensor, H=14, W=14)
    print(output.shape) # (1, 196, 32)
    ```

    """
    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    layer = layer_class(
        project_dim=project_dim, num_heads=num_heads, sr_ratio=sr_ratio
    )

    return layer
