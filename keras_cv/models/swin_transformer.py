import collections.abc
from functools import partial
from itertools import repeat
from typing import Dict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras_cv.transformers import BasicLayer
from keras_cv.layers import PatchMerging

# Take from here:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer.py
def swin_base_patch4_window12_384():
    """Swin-B @ 384x384, pretrained ImageNet-22k, fine tune 1k"""
    cfg = dict(
        img_size=384,
        patch_size=4,
        window_size=12,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        name="swin_base_patch4_window12_384",
    )
    return cfg


def swin_base_patch4_window7_224():
    """Swin-B @ 224x224, pretrained ImageNet-22k, fine tune 1k"""
    cfg = dict(
        patch_size=4,
        window_size=7,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        name="swin_base_patch4_window7_224",
    )
    return cfg


def swin_large_patch4_window12_384():
    """Swin-L @ 384x384, pretrained ImageNet-22k, fine tune 1k"""
    cfg = dict(
        img_size=384,
        patch_size=4,
        window_size=12,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        name="swin_large_patch4_window12_384",
    )
    return cfg


def swin_large_patch4_window7_224():
    """Swin-L @ 224x224, pretrained ImageNet-22k, fine tune 1k"""
    cfg = dict(
        patch_size=4,
        window_size=7,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        name="swin_large_patch4_window7_224",
    )
    return cfg


def swin_small_patch4_window7_224():
    """Swin-S @ 224x224, trained ImageNet-1k"""
    cfg = dict(
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        name="swin_small_patch4_window7_224",
    )
    return cfg


def swin_tiny_patch4_window7_224():
    """Swin-T @ 224x224, trained ImageNet-1k"""
    cfg = dict(
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        name="swin_tiny_patch4_window7_224",
    )
    return cfg


def swin_base_patch4_window12_384_in22k():
    """Swin-B @ 384x384, trained ImageNet-22k"""
    cfg = dict(
        img_size=384,
        patch_size=4,
        window_size=12,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        name="swin_base_patch4_window12_384_in22k",
        num_classes=21841,
    )
    return cfg


def swin_base_patch4_window7_224_in22k():
    """Swin-B @ 224x224, trained ImageNet-22k"""
    cfg = dict(
        patch_size=4,
        window_size=7,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        name="swin_base_patch4_window7_224_in22k",
        num_classes=21841,
    )
    return cfg


def swin_large_patch4_window12_384_in22k():
    """Swin-L @ 384x384, trained ImageNet-22k"""
    cfg = dict(
        img_size=384,
        patch_size=4,
        window_size=12,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        name="swin_large_patch4_window12_384_in22k",
        num_classes=21841,
    )
    return cfg


def swin_large_patch4_window7_224_in22k():
    """Swin-L @ 224x224, trained ImageNet-22k"""
    cfg = dict(
        patch_size=4,
        window_size=7,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        name="swin_large_patch4_window7_224_in22k",
        num_classes=21841,
    )
    return cfg


def swin_s3_tiny_224():
    """Swin-S3-T @ 224x224, ImageNet-1k. https://arxiv.org/abs/2111.14725"""
    cfg = dict(
        patch_size=4,
        window_size=(7, 7, 14, 7),
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        name="swin_s3_tiny_224",
    )
    return cfg


def swin_s3_small_224():
    """Swin-S3-S @ 224x224, trained ImageNet-1k. https://arxiv.org/abs/2111.14725"""
    cfg = dict(
        patch_size=4,
        window_size=(14, 14, 14, 7),
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        name="swin_s3_small_224",
    )
    return cfg


def swin_s3_base_224():
    """Swin-S3-B @ 224x224, trained ImageNet-1k. https://arxiv.org/abs/2111.14725"""
    cfg = dict(
        patch_size=4,
        window_size=(7, 7, 14, 7),
        embed_dim=96,
        depths=(2, 2, 30, 2),
        num_heads=(3, 6, 12, 24),
        name="swin_s3_base_224",
    )
    return cfg


MODEL_MAP = {
    "swin_base_patch4_window12_384": swin_base_patch4_window12_384,
    "swin_base_patch4_window7_224": swin_base_patch4_window7_224,
    "swin_large_patch4_window12_384": swin_large_patch4_window12_384,
    "swin_large_patch4_window7_224": swin_large_patch4_window7_224,
    "swin_small_patch4_window7_224": swin_small_patch4_window7_224,
    "swin_tiny_patch4_window7_224": swin_tiny_patch4_window7_224,
    "swin_base_patch4_window12_384_in22k": swin_base_patch4_window12_384_in22k,
    "swin_base_patch4_window7_224_in22k": swin_base_patch4_window7_224_in22k,
    "swin_large_patch4_window12_384_in22k": swin_large_patch4_window12_384_in22k,
    "swin_large_patch4_window7_224_in22k": swin_large_patch4_window7_224_in22k,
    "swin_s3_tiny_224": swin_s3_tiny_224,
    "swin_s3_small_224": swin_s3_small_224,
    "swin_s3_base_224": swin_s3_base_224,
}

"""
Code copied and modified from 
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer.py
"""

# https://github.com/rwightman/pytorch-image-models/blob/6d4665bb52390974e0cf9674c60c41946d2f4ee2/timm/models/layers/helpers.py#L10
def to_ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


class SwinTransformer(keras.Model):
    """Swin Transformer
        A TensorFlow impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        head_dim (int, tuple(int)):
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (layers.Layer): Normalization layer. Default: layers.LayerNormalization.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        pre_logits (bool): If True, return model without classification head. Default: False
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        num_classes=1000,
        global_pool="avg",
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        head_dim=None,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-5),
        ape=False,
        patch_norm=True,
        pre_logits=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.img_size = (
            img_size
            if isinstance(img_size, collections.abc.Iterable)
            else (img_size, img_size)
        )
        self.patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.ape = ape

        # split image into non-overlapping patches
        self.projection = keras.Sequential(
            [
                layers.Conv2D(
                    filters=embed_dim,
                    kernel_size=(patch_size, patch_size),
                    strides=(patch_size, patch_size),
                    padding="VALID",
                    name="conv_projection",
                    kernel_initializer="lecun_normal",
                ),
                layers.Reshape(
                    target_shape=(-1, embed_dim),
                    name="flatten_projection",
                ),
            ],
            name="projection",
        )
        if patch_norm:
            self.projection.add(norm_layer())

        self.patch_grid = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.patch_grid[0] * self.patch_grid[1]

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = tf.Variable(
                tf.zeros((1, self.num_patches, self.embed_dim)),
                trainable=True,
                name="absolute_pos_embed",
            )
        else:
            self.absolute_pos_embed = None
        self.pos_drop = layers.Dropout(drop_rate)

        # build layers
        if not isinstance(self.embed_dim, (tuple, list)):
            self.embed_dim = [
                int(self.embed_dim * 2 ** i) for i in range(self.num_layers)
            ]
        embed_out_dim = self.embed_dim[1:] + [None]
        head_dim = to_ntuple(self.num_layers)(head_dim)
        window_size = to_ntuple(self.num_layers)(window_size)
        mlp_ratio = to_ntuple(self.num_layers)(mlp_ratio)
        dpr = [
            float(x) for x in tf.linspace(0.0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        swin_layers = [
            BasicLayer(
                dim=self.embed_dim[i],
                out_dim=embed_out_dim[i],
                input_resolution=(
                    self.patch_grid[0] // (2 ** i),
                    self.patch_grid[1] // (2 ** i),
                ),
                depth=depths[i],
                num_heads=num_heads[i],
                head_dim=head_dim[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio[i],
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i < self.num_layers - 1) else None,
                name=f"basic_layer_{i}",
            )
            for i in range(self.num_layers)
        ]
        self.swin_layers = swin_layers

        self.norm = norm_layer()

        self.pre_logits = pre_logits
        if not self.pre_logits:
            self.head = layers.Dense(num_classes, name="classification_head")

    def forward_features(self, x):
        x = self.projection(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for swin_layer in self.swin_layers:
            x = swin_layer(x)

        x = self.norm(x)  # [B, L, C]
        return x

    def forward_head(self, x):
        if self.global_pool == "avg":
            x = tf.reduce_mean(x, axis=1)
        return x if self.pre_logits else self.head(x)

    def call(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    # Thanks to Willi Gierke for this suggestion.
    @tf.function(
        input_signature=[tf.TensorSpec([None, None, None, 3], tf.float32)]
    )
    def get_attention_scores(
        self, x: tf.Tensor
    ) -> Dict[str, Dict[str, tf.Tensor]]:
        all_attention_scores = {}

        x = self.projection(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for i, swin_layer in enumerate(self.swin_layers):
            x, attention_scores = swin_layer(x, return_attns=True)
            all_attention_scores.update({f"swin_stage_{i}": attention_scores})

        return all_attention_scores