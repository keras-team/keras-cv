from keras_cv.backend import keras

"""
Based on: https://github.com/sithu31296/semantic-segmentation/blob/main/semseg/models/backbones/mit.py
"""


@keras.saving.register_keras_serializable(package="keras_cv")
class EfficientMultiheadAttention(tf.keras.layers.Layer):
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

    def call(self, x, H, W):
        input_shape = shape(x)

        q = self.q(x)
        q = q.reshape(
            (
                input_shape[0],
                input_shape[1],
                self.num_heads,
                input_shape[2] // self.num_heads,
            ),
        )

        q = q.transpose([0, 2, 1, 3])

        if self.sr_ratio > 1:
            x = x.transpose(x, [0, 2, 1]).reshape(
                (input_shape[0], H, W, input_shape[2])
            )
            x = self.sr(x)
            x = x.reshape([input_shape[0], input_shape[2], -1])
            x = x.transpose([0, 2, 1])
            x = self.norm(x)

        k = self.k(x)
        v = self.v(x)

        k = k.transpose([0, 2, 1, 3]).reshape(
            [
                input_shape[0],
                -1,
                self.num_heads,
                input_shape[2] // self.num_heads,
            ]
        )
        v = (
            v.transpose([0, 2, 1, 3]).reshape(
                [
                    input_shape[0],
                    -1,
                    self.num_heads,
                    input_shape[2] // self.num_heads,
                ]
            ),
        )

        attn = (q @ x.transpose([0, 1, 3, 2])) * self.scale
        attn = keras.nn.ops.softmax(attn, axis=-1)

        attn = attn @ v
        attn = attn.transpose(attn, [0, 2, 1, 3]).reshape(
            [input_shape[0], input_shape[1], input_shape[2]]
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
