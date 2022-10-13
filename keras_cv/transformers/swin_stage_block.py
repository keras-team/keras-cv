"""
Code copied and modified from
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer.py
"""

from functools import partial
from typing import Dict, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

from keras_cv.transformers.swin_block import SwinTransformerBlock


class BasicLayer(keras.Model):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        head_dim (int): Channels per head (dim // num_heads if not set)
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | list[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (layers.Layer, optional): Normalization layer. Default: layers.LayerNormalization
        downsample (layers.Layer | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(
        self,
        dim,
        out_dim,
        input_resolution,
        depth,
        num_heads=4,
        head_dim=None,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=partial(L.LayerNormalization, epsilon=1e-5),
        downsample=None,
        **kwargs,
    ):

        super().__init__(kwargs)

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        blocks = [
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                head_dim=head_dim,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, list)
                else drop_path,
                norm_layer=norm_layer,
                name=f"swin_transformer_block_{i}",
            )
            for i in range(depth)
        ]
        self.blocks = blocks

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution,
                dim=dim,
                out_dim=out_dim,
                norm_layer=norm_layer,
            )
        else:
            self.downsample = None

    def call(
        self, x, return_attns=False
    ) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        if return_attns:
            attention_scores = {}

        for i, block in enumerate(self.blocks):
            if not return_attns:
                x = block(x)
            else:
                x, attns = block(x, return_attns)
                attention_scores.update({f"swin_block_{i}": attns})
        if self.downsample is not None:
            x = self.downsample(x)

        if return_attns:
            return x, attention_scores
        else:
            return x