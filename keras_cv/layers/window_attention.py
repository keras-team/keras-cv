"""
Code copied and modified from
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer.py
https://github.com/sayakpaul/swin-transformers-tf/blob/main/swins/layers/window_attn.py
"""


import collections.abc
from typing import Tuple, Union

import tensorflow as tf
from tensorflow.keras import layers


def get_relative_position_index(win_h, win_w):
    # get pair-wise relative position index for each token inside the window
    xx, yy = tf.meshgrid(range(win_h), range(win_w))
    coords = tf.stack([yy, xx], axis=0)  # [2, Wh, Ww]
    coords_flatten = tf.reshape(coords, [2, -1])  # [2, Wh*Ww]

    relative_coords = (
        coords_flatten[:, :, None] - coords_flatten[:, None, :]
    )  # [2, Wh*Ww, Wh*Ww]
    relative_coords = tf.transpose(
        relative_coords, perm=[1, 2, 0]
    )  # [Wh*Ww, Wh*Ww, 2]

    xx = (relative_coords[:, :, 0] + win_h - 1) * (2 * win_w - 1)
    yy = relative_coords[:, :, 1] + win_w - 1
    relative_coords = tf.stack([xx, yy], axis=-1)

    return tf.reduce_sum(relative_coords, axis=-1)  # [Wh*Ww, Wh*Ww]


class WindowAttention(layers.Layer):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        head_dim (int): Number of channels per head (dim // num_heads if not set)
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        num_heads,
        head_dim=None,
        window_size=7,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.dim = dim
        self.window_size = (
            window_size
            if isinstance(window_size, collections.abc.Iterable)
            else (window_size, window_size)
        )  # Wh, Ww
        self.win_h, self.win_w = self.window_size
        self.window_area = self.win_h * self.win_w
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.attn_dim = self.head_dim * num_heads
        self.scale = self.head_dim ** -0.5

        # get pair-wise relative position index for each token inside the window
        self.relative_position_index = get_relative_position_index(
            self.win_h, self.win_w
        )

        self.qkv = layers.Dense(
            self.attn_dim * 3, use_bias=qkv_bias, name="attention_qkv"
        )
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim, name="attention_projection")
        self.proj_drop = layers.Dropout(proj_drop)

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(
            shape=((2 * self.win_h - 1) * (2 * self.win_w - 1), self.num_heads),
            initializer="zeros",
            trainable=True,
            name="relative_position_bias_table",
        )
        super().build(input_shape)

    def _get_rel_pos_bias(self) -> tf.Tensor:
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            self.relative_position_index,
            axis=0,
        )
        return tf.transpose(relative_position_bias, [2, 0, 1])

    def call(
        self, x, mask=None, return_attns=False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (B_, N, 3, self.num_heads, -1))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))

        q, k, v = tf.unstack(qkv, 3)

        scale = tf.cast(self.scale, dtype=qkv.dtype)
        q = q * scale
        attn = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2]))
        attn = attn + self._get_rel_pos_bias()

        if mask is not None:
            num_win = tf.shape(mask)[0]
            attn = tf.reshape(
                attn, (B_ // num_win, num_win, self.num_heads, N, N)
            )
            attn = attn + tf.expand_dims(mask, 1)[None, ...]

            attn = tf.reshape(attn, (-1, self.num_heads, N, N))
            attn = tf.nn.softmax(attn, -1)
        else:
            attn = tf.nn.softmax(attn, -1)

        attn = self.attn_drop(attn)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (B_, N, C))

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attns:
            return x, attn
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "win_h": self.win_h,
                "win_w": self.win_w,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "attn_dim": self.attn_dim,
                "scale": self.scale,
            }
        )
        return config