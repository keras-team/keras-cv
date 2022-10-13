"""
Code copied and modified from
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer.py
https://github.com/sayakpaul/swin-transformers-tf/blob/main/swins/layers/patch_merging.py
"""

from functools import partial

import tensorflow as tf
from tensorflow.keras import layers

class PatchMerging(layers.Layer):
    """Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
    """

    def __init__(
        self,
        input_resolution,
        dim,
        out_dim=None,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-5),
        **kwargs
    ):
        super().__init__(**kwargs)

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer()
        self.reduction = layers.Dense(self.out_dim, use_bias=False)

    def call(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        x = tf.reshape(x, (B, H, W, C))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = tf.concat([x0, x1, x2, x3], axis=-1)  # [B, H/2, W/2, 4*C]
        x = tf.reshape(x, (B, -1, 4 * C))  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_resolution": self.input_resolution,
                "dim": self.dim,
                "out_dim": self.out_dim,
                "norm": self.norm,
            }
        )
        return config