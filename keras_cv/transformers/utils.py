"""
Code copied and modified from
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer.py
"""

import tensorflow as tf


def window_partition_swin(x: tf.Tensor, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

    x = tf.reshape(
        x, (B, H // window_size, window_size, W // window_size, window_size, C)
    )
    windows = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(windows, (-1, window_size, window_size, C))
    return windows


def window_reverse_swin(windows: tf.Tensor, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = tf.shape(windows)[0] // tf.cast(
        H * W / window_size / window_size, dtype="int32"
    )

    x = tf.reshape(
        windows,
        (
            B,
            H // window_size,
            W // window_size,
            window_size,
            window_size,
            -1,
        ),
    )
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, (B, H, W, -1))