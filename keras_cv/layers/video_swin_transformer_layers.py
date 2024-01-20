# Copyright 2023 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.layers import DropPath

import numpy as np
from functools import partial
from keras import layers

def window_partition(x, window_size):
    """
    Args:
        x: (batch_size, depth, height, width, channel)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """

    input_shape = ops.shape(x)
    batch_size, depth, height, width, channel = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
        input_shape[4],
    )

    x = ops.reshape(
        x,
        [
            batch_size,
            depth // window_size[0],
            window_size[0],
            height // window_size[1],
            window_size[1],
            width // window_size[2],
            window_size[2],
            channel,
        ],
    )

    x = ops.transpose(x, [0, 1, 3, 5, 2, 4, 6, 7])
    windows = ops.reshape(
        x, [-1, window_size[0] * window_size[1] * window_size[2], channel]
    )

    return windows


def window_reverse(windows, window_size, batch_size, depth, height, width):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        height (int): Height of image
        width (int): Width of image

    Returns:
        x: (batch_size, depth, height, width, channel)
    """
    x = ops.reshape(
        windows,
        [
            batch_size,
            depth // window_size[0],
            height // window_size[1],
            width // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        ],
    )
    x = ops.transpose(x, [0, 1, 4, 2, 5, 3, 6, 7])
    x = ops.reshape(x, [batch_size, depth, height, width, -1])
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)

    if shift_size is not None:
        use_shift_size = list(shift_size)

    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


def compute_mask(depth, height, width, window_size, shift_size):
    img_mask = np.zeros((1, depth, height, width, 1))
    cnt = 0
    for d in (
        slice(-window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    ):
        for h in (
            slice(-window_size[1]),
            slice(-window_size[1], -shift_size[1]),
            slice(-shift_size[1], None),
        ):
            for w in (
                slice(-window_size[2]),
                slice(-window_size[2], -shift_size[2]),
                slice(-shift_size[2], None),
            ):
                img_mask[:, d, h, w, :] = cnt
                cnt = cnt + 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = ops.squeeze(mask_windows, axis=-1)
    attn_mask = ops.expand_dims(mask_windows, axis=1) - ops.expand_dims(
        mask_windows, axis=2
    )
    attn_mask = ops.where(attn_mask != 0, -100.0, attn_mask)
    attn_mask = ops.where(attn_mask == 0, 0.0, attn_mask)
    return attn_mask


class MLP(layers.Layer):
    """A Multilayer perceptron(MLP) layer.

    Args:
        hidden_dim (int): The number of units in the hidden layers.
        output_dim (int): The number of units in the output layer.
        drop_rate  (float): Float between 0 and 1. Fraction of the 
            input units to drop.
        activation (str): Activation to use in the hidden layers.
            Default is `"gelu"`.

    References:
        - [Video Swin Transformer](https://arxiv.org/abs/2106.13230)
        - [Video Swin Transformer GitHub](https://github.com/SwinTransformer/Video-Swin-Transformer)
    """ # noqa: E501

    def __init__(
        self,
        hidden_dim,
        output_dim,
        drop_rate,
        activation="gelu",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.drop_rate = drop_rate
        self.activation = activation
        self.fc1 = layers.Dense(self.hidden_dim)
        self.fc2 = layers.Dense(self.output_dim)
        self.dropout = layers.Dropout(self.drop_rate)

    def call(self, x, training=None):
        x = self.fc1(x)
        x = layers.Activation(self.activation)(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim, 
                "hidden_dim": self.hidden_dim,
                "drop_rate": self.drop_rate,
                "activation": self.activation
            }
        )
        return config
    

class PatchEmbedding3D(keras.Model):
    """Video to Patch Embedding layer.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (keras.layers, optional): Normalization layer. Default: None

    References:
        - [Video Swin Transformer](https://arxiv.org/abs/2106.13230)
        - [Video Swin Transformer GitHub](https://github.com/SwinTransformer/Video-Swin-Transformer)
    """ # noqa: E501

    def __init__(self, patch_size=(2, 4, 4), embed_dim=96, norm_layer=None, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer

        self.proj = layers.Conv3D(
            self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            name="embed_proj",
        )
        if self.norm_layer is not None:
            self.norm = self.norm_layer(axis=-1, epsilon=1e-5, name="embed_norm")
        else:
            self.norm = None

    def build(self, input_shape):
        self.pads = [
            [0, 0],
            self._compute_padding(input_shape[1], self.patch_size[0]),
            self._compute_padding(input_shape[2], self.patch_size[1]),
            self._compute_padding(input_shape[3], self.patch_size[2]),
            [0, 0],
        ]

    def _compute_padding(self, dim, patch_size):
        pad_amount = patch_size - (dim % patch_size)
        return [0, pad_amount if pad_amount != patch_size else 0]

    def compute_output_shape(self, input_shape):
        spatial_dims = [
            (dim - self.patch_size[i]) // self.patch_size[i] + 1
            for i, dim in enumerate(input_shape[1:-1])
        ]
        output_shape = (input_shape[0],) + tuple(spatial_dims) + (self.embed_dim,)
        return output_shape

    def call(self, x):
        x = ops.pad(x, self.pads)
        x = self.proj(x)

        if self.norm is not None:
            x = self.norm(x)

        return x
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size, 
                "embed_dim": self.embed_dim,
            }
        )
        return config
    

class PatchMerging(layers.Layer):
    """Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (keras.layers, optional): Normalization layer.  Default: LayerNormalization

    References:
        - [Video Swin Transformer](https://arxiv.org/abs/2106.13230)
        - [Video Swin Transformer GitHub](https://github.com/SwinTransformer/Video-Swin-Transformer)
    """ # noqa: E501

    def __init__(self, dim, norm_layer=layers.LayerNormalization, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.reduction = layers.Dense(2 * dim, use_bias=False)
        self.norm = norm_layer(axis=-1, epsilon=1e-5)

    def call(self, x):
        input_shape = ops.shape(x)
        height, width = (
            input_shape[2],
            input_shape[3],
        )

        # padding if needed
        paddings = [
            [0, 0],
            [0, 0],
            [0, ops.mod(height, 2)],
            [0, ops.mod(width, 2)],
            [0, 0],
        ]
        x = ops.pad(x, paddings)
        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = ops.concatenate([x0, x1, x2, x3], axis=-1)  # B D H/2 W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
            }
        )
        return config
    

@keras_cv_export(
    "keras_cv.layers.WindowAttention3D", package="keras_cv.layers"
)
class WindowAttention3D(keras.Model):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0

    References:
        - [Video Swin Transformer](https://arxiv.org/abs/2106.13230)
        - [Video Swin Transformer GitHub](https://github.com/SwinTransformer/Video-Swin-Transformer)
    """ # noqa: E501

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias,
        qk_scale = None,
        attn_drop_rate = 0.0,
        proj_drop_rate = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        # variables
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv_bias = qkv_bias
        self.attn_drop_rate = attn_drop_rate
        self.proj_drop_rate = proj_drop_rate

        self.relative_position_bias_table = self.add_weight(
            shape=(
                (2 * self.window_size[0] - 1)
                * (2 * self.window_size[1] - 1)
                * (2 * self.window_size[2] - 1),
                self.num_heads,
            ),
            initializer="zeros",
            trainable=True,
            name="relative_position_bias_table",
        )
        self.relative_position_index = self.get_relative_position_index(
            self.window_size[0], self.window_size[1], self.window_size[2]
        )

        # layers
        self.qkv = layers.Dense(self.dim * 3, use_bias=self.qkv_bias)
        self.attn_drop = layers.Dropout(self.attn_drop_rate)
        self.proj = layers.Dense(self.dim)
        self.proj_drop = layers.Dropout(self.proj_drop_rate)

    def get_relative_position_index(self, window_depth, window_height, window_width):
        y_y, z_z, x_x = ops.meshgrid(
            range(window_width), range(window_depth), range(window_height)
        )
        coords = ops.stack([z_z, y_y, x_x], axis=0)
        coords_flatten = ops.reshape(coords, [3, -1])
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = ops.transpose(relative_coords, [1, 2, 0])
        z_z = (
            (relative_coords[:, :, 0] + window_depth - 1)
            * (2 * window_height - 1)
            * (2 * window_width - 1)
        )
        x_x = (relative_coords[:, :, 1] + window_height - 1) * (2 * window_width - 1)
        y_y = relative_coords[:, :, 2] + window_width - 1
        relative_coords = ops.stack([z_z, x_x, y_y], axis=-1)
        return ops.sum(relative_coords, axis=-1)

    def call(self, x, mask=None, return_attention_maps=False, training=None):
        input_shape = ops.shape(x)
        batch_size, depth, channel = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )

        qkv = self.qkv(x)
        qkv = ops.reshape(
            qkv, [batch_size, depth, 3, self.num_heads, channel // self.num_heads]
        )
        qkv = ops.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = ops.split(qkv, 3, axis=0)

        q = ops.squeeze(q, axis=0) * self.scale
        k = ops.squeeze(k, axis=0)
        v = ops.squeeze(v, axis=0)
        attention_maps = ops.matmul(q, ops.transpose(k, [0, 1, 3, 2]))

        relative_position_bias = ops.take(
            self.relative_position_bias_table,
            self.relative_position_index[:depth, :depth],
        )
        relative_position_bias = ops.reshape(relative_position_bias, [depth, depth, -1])
        relative_position_bias = ops.transpose(relative_position_bias, [2, 0, 1])
        attention_maps = attention_maps + relative_position_bias[None, ...]

        if mask is not None:
            mask_size = ops.shape(mask)[0]
            mask = ops.cast(mask, dtype=attention_maps.dtype)
            attention_maps = ops.reshape(
                attention_maps,
                [batch_size // mask_size, mask_size, self.num_heads, depth, depth],
            )
            attention_maps = attention_maps + mask[:, None, :, :]
            attention_maps = ops.reshape(
                attention_maps, [-1, self.num_heads, depth, depth]
            )

        attention_maps = keras.activations.softmax(attention_maps, axis=-1)
        attention_maps = self.attn_drop(attention_maps, training=training)

        x = ops.matmul(attention_maps, v)
        x = ops.transpose(x, [0, 2, 1, 3])
        x = ops.reshape(x, [batch_size, depth, channel])
        x = self.proj(x)
        x = self.proj_drop(x, training=training)

        if return_attention_maps:
            return x, attention_maps
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "scale": self.scale,
                "qkv_bias": self.qkv_bias,
                "attn_drop_rate": self.attn_drop_rate,
                "proj_drop_rate": self.proj_drop_rate,
            }
        )
        return config
    

@keras_cv_export(
    "keras_cv.layers.SwinTransformerBlock3D", package="keras_cv.layers"
)
class SwinTransformerBlock3D(keras.Model):
    """Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (keras.layers.Activation, optional): Activation layer. Default: gelu
        norm_layer (keras.layers, optional): Normalization layer.  Default: LayerNormalization
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size = (2, 7, 7),
        shift_size = (0, 0, 0),
        mlp_ratio = 4.0,
        qkv_bias = True,
        qk_scale = None,
        drop_rate = 0.0,
        attn_drop = 0.0,
        drop_path = 0.0,
        activation = "gelu",
        norm_layer = layers.LayerNormalization,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.drop_rate = drop_rate
        self.drop_path = drop_path
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.activation = activation
        self.norm_layer = norm_layer

        assert (
            0 <= self.shift_size[0] < self.window_size[0]
        ), "shift_size must in 0-window_size"
        assert (
            0 <= self.shift_size[1] < self.window_size[1]
        ), "shift_size must in 0-window_size"
        assert (
            0 <= self.shift_size[2] < self.window_size[2]
        ), "shift_size must in 0-window_size"

    def build(self, input_shape):
        self.window_size, self.shift_size = get_window_size(
            input_shape[1:-1], self.window_size, self.shift_size
        )
        if any(i > 0 for i in self.shift_size):
            self.roll = True
        else:
            self.roll = False

        # layers
        self.norm1 = self.norm_layer(axis=-1, epsilon=1e-05)
        self.attn = WindowAttention3D(
            self.dim,
            window_size=self.window_size,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            attn_drop=self.attn_drop,
            proj_drop=self.drop_rate,
        )
        self.drop_path = (
            DropPath(self.drop_path) if self.drop_path > 0.0 else layers.Identity()
        )
        self.norm2 = self.norm_layer(axis=-1, epsilon=1e-05)
        self.mlp = MLP(
            output_dim=self.dim,
            hidden_dim=self.mlp_hidden_dim,
            activation=self.activation,
            drop_rate=self.drop_rate,
        )

    def _forward(self, x, mask_matrix, return_attention_maps, training):
        input_shape = ops.shape(x)
        batch_size, depth, height, width, channel = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
            input_shape[4],
        )
        window_size, shift_size = self.window_size, self.shift_size
        x = self.norm1(x)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = ops.mod(-depth + window_size[0], window_size[0])
        pad_b = ops.mod(-height + window_size[1], window_size[1])
        pad_r = ops.mod(-width + window_size[2], window_size[2])
        paddings = [[0, 0], [pad_d0, pad_d1], [pad_t, pad_b], [pad_l, pad_r], [0, 0]]
        x = ops.pad(x, paddings)

        input_shape = ops.shape(x)
        depth_p, height_p, width_p = (
            input_shape[1],
            input_shape[2],
            input_shape[3],
        )

        # Cyclic Shift
        if self.roll:
            shifted_x = ops.roll(
                x,
                shift=(-shift_size[0], -shift_size[1], -shift_size[2]),
                axis=(1, 2, 3),
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)

        # get attentions params
        if return_attention_maps:
            attention_windows, attention_maps = self.attn(
                x_windows,
                mask=attn_mask,
                return_attention_maps=return_attention_maps,
                training=training,
            )
        else:
            attention_windows = self.attn(x_windows, mask=attn_mask, training=training)

        # reverse the swin windows
        shifted_x = window_reverse(
            attention_windows, window_size, batch_size, depth_p, height_p, width_p
        )

        # Reverse Cyclic Shift
        if self.roll:
            x = ops.roll(
                shifted_x,
                shift=(shift_size[0], shift_size[1], shift_size[2]),
                axis=(1, 2, 3),
            )
        else:
            x = shifted_x

        # pad if required
        do_pad = ops.logical_or(
            ops.greater(pad_d1, 0),
            ops.logical_or(ops.greater(pad_r, 0), ops.greater(pad_b, 0)),
        )
        x = ops.cond(do_pad, lambda: x[:, :depth, :height, :width, :], lambda: x)

        if return_attention_maps:
            return x, attention_maps

        return x

    def call(self, x, mask_matrix=None, return_attention_maps=False, training=None):
        shortcut = x
        x = self._forward(x, mask_matrix, return_attention_maps, training)

        if return_attention_maps:
            x, attention_maps = x

        x = shortcut + self.drop_path(x)
        x = self.drop_path(self.mlp(self.norm2(x)), training=training)

        if return_attention_maps:
            return x, attention_maps

        return x
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.num_heads,
                "num_heads": self.window_size,
                "shift_size": self.shift_size,
                "mlp_ratio": self.mlp_ratio,
                "qkv_bias": self.qkv_bias,
                "qk_scale": self.qk_scale,
                "attn_drop": self.attn_drop,
                "drop_rate": self.drop_rate, 
                "drop_path": self.drop_path,
                "mlp_hidden_dim": self.mlp_hidden_dim,
                "activation": self.activation
            }
        )
        return config


class BasicLayer(keras.Model):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (keras.layers, optional): Normalization layer. Default: LayerNormalization
        downsample (keras.layers | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size = (1, 7, 7),
        mlp_ratio = 4.0,
        qkv_bias = False,
        qk_scale = None,
        drop_rate = 0.0,
        attn_drop = 0.0,
        drop_path = 0.0,
        norm_layer = partial(
            layers.LayerNormalization, epsilon=1e-05
        ),
        downsample = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.shift_size = tuple([i // 2 for i in window_size])
        self.depth = depth
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop = attn_drop
        self.drop_path = drop_path
        self.norm_layer = norm_layer
        self.downsample = downsample

    def compute_dim_padded(self, input_dim, window_dim_size):
        input_dim = ops.cast(input_dim, dtype="float32")
        window_dim_size = ops.cast(window_dim_size, dtype="float32")
        return ops.cast(
            ops.ceil(input_dim / window_dim_size) * window_dim_size, "int32"
        )

    def compute_output_shape(self, input_shape):
        window_size, _ = get_window_size(
            input_shape[1:-1], self.window_size, self.shift_size
        )
        depth_p = self.compute_dim_padded(input_shape[1], window_size[0])
        height_p = self.compute_dim_padded(input_shape[2], window_size[1])
        width_p = self.compute_dim_padded(input_shape[3], window_size[2])
        output_shape = (input_shape[0], depth_p, height_p, width_p, self.dim)
        return output_shape

    def build(self, input_shape):
        window_size, shift_size = get_window_size(
            input_shape[1:-1], self.window_size, self.shift_size
        )
        depth_p = self.compute_dim_padded(input_shape[1], window_size[0])
        height_p = self.compute_dim_padded(input_shape[2], window_size[1])
        width_p = self.compute_dim_padded(input_shape[3], window_size[2])
        self.attn_mask = compute_mask(
            depth_p, height_p, width_p, window_size, shift_size
        )

        # build blocks
        self.blocks = [
            SwinTransformerBlock3D(
                dim=self.dim,
                num_heads=self.num_heads,
                window_size=self.window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop_rate=self.drop_rate,
                attn_drop=self.attn_drop,
                drop_path=self.drop_path[i]
                if isinstance(self.drop_path, list)
                else self.drop_path,
                norm_layer=self.norm_layer,
            )
            for i in range(self.depth)
        ]

        if self.downsample is not None:
            self.downsample = self.downsample(dim=self.dim, norm_layer=self.norm_layer)

    def call(self, x, training=None, return_attention_maps=False):
        input_shape = ops.shape(x)
        batch_size, depth, height, width, channel = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
            input_shape[4],
        )

        for block in self.blocks:
            if return_attention_maps:
                x, attention_maps = block(
                    x,
                    self.attn_mask,
                    return_attention_maps=return_attention_maps,
                    training=training,
                )
            else:
                x = block(x, self.attn_mask, training=training)

        x = ops.reshape(x, [batch_size, depth, height, width, -1])

        if self.downsample is not None:
            x = self.downsample(x)

        if return_attention_maps:
            return x, attention_maps

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "shift_size": self.shift_size,
                "depth": self.depth,
                "qkv_bias": self.qkv_bias,
                "qk_scale": self.qk_scale,
                "drop": self.drop,
                "attn_drop": self.attn_drop,
                "drop_path": self.drop_path,
            }
        )
        return config