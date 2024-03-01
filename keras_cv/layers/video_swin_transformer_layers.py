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

from functools import partial

import numpy as np
from keras import layers

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.layers import DropPath


def window_partition(x, window_size):
    """
    Args:
        x: (batch_size, depth, height, width, channel)
        window_size (tuple[int]): window size

    Returns:
        windows: (batch_size*num_windows, window_size*window_size, channel)
    """  # noqa: E501

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
        windows: (batch_size*num_windows, window_size, window_size, channel)
        window_size (tuple[int]): Window size
        height (int): Height of image
        width (int): Width of image

    Returns:
        x: (batch_size, depth, height, width, channel)
    """  # noqa: E501
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

    Returns:
        x: window_size, shift_size
    """  # noqa: E501

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
        hidden_dim (int): The number of units in the hidden layer.
        output_dim (int): The number of units in the output layer.
        drop_rate  (float): Float between 0 and 1. Fraction of the
            input units to drop.
        activation (str): Activation to use in the hidden layers.
            Default is `"gelu"`.

    References:
        - [Video Swin Transformer](https://arxiv.org/abs/2106.13230)
        - [Video Swin Transformer GitHub](https://github.com/SwinTransformer/Video-Swin-Transformer)
    """  # noqa: E501

    def __init__(
        self,
        hidden_dim,
        output_dim,
        drop_rate=0.0,
        activation='gelu',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self._activation_identifier = activation
        self.drop_rate = drop_rate
        self.activation = layers.Activation(self._activation_identifier)
        self.fc1 = layers.Dense(self.hidden_dim)
        self.fc2 = layers.Dense(self.output_dim)
        self.dropout = layers.Dropout(self.drop_rate)
        
    def build(self, input_shape):
        self.fc1.build(input_shape)
        self.fc2.build((*input_shape[1:-1], self.hidden_dim))
        self.built = True

    def call(self, x, training=None):
        x = self.fc1(x)
        x = self.activation(x)
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
                'activation': self._activation_identifier
            }
        )
        return config
    

class PatchEmbed3D(keras.Model):
    """Video to Patch Embedding layer.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (keras.layers, optional): Normalization layer. Default: None

    References:
        - [Video Swin Transformer](https://arxiv.org/abs/2106.13230)
        - [Video Swin Transformer GitHub](https://github.com/SwinTransformer/Video-Swin-Transformer)
    """  # noqa: E501
        
    def __init__(
        self, 
        patch_size=(2, 4, 4), 
        embed_dim=96, 
        norm_layer=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer

    def _compute_padding(self, dim, patch_size):
        pad_amount = patch_size - (dim % patch_size)
        return [
            0, pad_amount if pad_amount != patch_size else 0
        ]

    def build(self, input_shape):
        self.pads = [
            [0, 0], 
            self._compute_padding(input_shape[1], self.patch_size[0]),
            self._compute_padding(input_shape[2], self.patch_size[1]),
            self._compute_padding(input_shape[3], self.patch_size[2]),
            [0, 0]   
        ]
        
        self.proj = layers.Conv3D(
            self.embed_dim, 
            kernel_size=self.patch_size,
            strides=self.patch_size, 
            name='embed_proj'
        )
        self.proj.build((None, None, None, None, input_shape[-1]))
        
        self.norm = None
        if self.norm_layer is not None:
            self.norm = self.norm_layer(
                axis=-1, epsilon=1e-5, name='embed_norm'
            )
            self.norm.build(
                (None, None, None, None, self.embed_dim)
            )
        self.built = True

    def call(self, x):
        x = ops.pad(x, self.pads)
        x = self.proj(x)
        
        if self.norm is not None:
            x = self.norm(x) 
            
        return x
    
    def compute_output_shape(self, input_shape):
        spatial_dims = [
            (dim - self.patch_size[i]) // self.patch_size[i] + 1
            for i, dim in enumerate(input_shape[1:-1])
        ]
        output_shape = (input_shape[0],) + tuple(spatial_dims) + (self.embed_dim,)
        return output_shape

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
        input_dim (int): Number of input channels.
        norm_layer (keras.layers, optional): Normalization layer. 
            Default: LayerNormalization

    References:
        - [Video Swin Transformer](https://arxiv.org/abs/2106.13230)
        - [Video Swin Transformer GitHub](https://github.com/SwinTransformer/Video-Swin-Transformer)
    """  # noqa: E501
        
    def __init__(
        self, 
        input_dim, 
        norm_layer=layers.LayerNormalization,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.norm_layer = norm_layer

    def build(self, input_shape):
        batch_size, depth, height, width, channel = input_shape
        self.reduction = layers.Dense(2 * self.input_dim, use_bias=False)
        self.reduction.build((batch_size, depth, height // 2, width // 2, 4 * channel))
        self.norm = self.norm_layer(axis=-1, epsilon=1e-5)
        self.norm.build((batch_size, depth, height // 2, width // 2, 4 * channel))
        self.built=True
        
    def call(self, x):
        """ The call function.

        Args:
            x: Input feature, shape: (batch_size, depth, height, width, channel).
        """
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
            [0, 0]
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
                "input_dim": self.input_dim,
            }
        )
        return config
    

class WindowAttention3D(keras.Model):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop_rate (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.0

    References:
        - [Video Swin Transformer](https://arxiv.org/abs/2106.13230)
        - [Video Swin Transformer GitHub](https://github.com/SwinTransformer/Video-Swin-Transformer)
    """  # noqa: E501
        
    def __init__(
        self, 
        input_dim, 
        window_size, 
        num_heads, 
        qkv_bias=True, 
        qk_scale=None, 
        attn_drop_rate=0., 
        proj_drop_rate=0.,
        **kwargs
    ):
        super().__init__(**kwargs)
        # variables
        self.input_dim = input_dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = input_dim // num_heads
        self.qk_scale = qk_scale
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv_bias = qkv_bias
        self.attn_drop_rate = attn_drop_rate
        self.proj_drop_rate = proj_drop_rate

    def get_relative_position_index(self, window_depth, window_height, window_width):
        y_y, z_z, x_x = ops.meshgrid(
            ops.arange(window_width), ops.arange(window_depth), ops.arange(window_height)
        )
        coords = ops.stack([z_z, y_y, x_x], axis=0)
        coords_flatten = ops.reshape(coords, [3, -1])
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = ops.transpose(relative_coords, axes=[1, 2, 0])
        z_z = (relative_coords[:, :, 0] + window_depth  - 1) * (2 * window_height - 1) * (2 * window_width - 1)
        x_x = (relative_coords[:, :, 1] + window_height - 1) * (2 * window_width - 1)
        y_y = (relative_coords[:, :, 2] + window_width  - 1)
        relative_coords = ops.stack([z_z, x_x, y_y], axis=-1)
        return ops.sum(relative_coords, axis=-1)

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(
            shape=(
                (2 * self.window_size[0] - 1) * 
                (2 * self.window_size[1] - 1) * 
                (2 * self.window_size[2] - 1),
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
        self.qkv = layers.Dense(self.input_dim * 3, use_bias=self.qkv_bias)
        self.attn_drop = layers.Dropout(self.attn_drop_rate)
        self.proj = layers.Dense(self.input_dim)
        self.proj_drop = layers.Dropout(self.proj_drop_rate)
        self.qkv.build(input_shape)
        self.proj.build(input_shape) 
        self.built = True

    def call(self, x, mask=None, training=None):
        input_shape = ops.shape(x)
        batch_size, depth, channel = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )
        
        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, [batch_size, depth, 3, self.num_heads, channel // self.num_heads])
        qkv = ops.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = ops.split(qkv, 3, axis=0)

        q = ops.squeeze(q, axis=0) * self.scale
        k = ops.squeeze(k, axis=0)
        v = ops.squeeze(v, axis=0)
        attn = ops.matmul(q, ops.transpose(k, [0, 1, 3, 2]))
        
        rel_pos_bias = ops.take(
            self.relative_position_bias_table, self.relative_position_index[:depth, :depth]
        )
        rel_pos_bias = ops.reshape(rel_pos_bias, [depth, depth, -1])
        rel_pos_bias = ops.transpose(rel_pos_bias, [2, 0, 1])
        attn = attn + rel_pos_bias[None, ...]
  
        if mask is not None:
            mask_size = ops.shape(mask)[0]
            mask = ops.cast(mask, dtype=attn.dtype)
            attn = ops.reshape(
                attn, 
                [batch_size // mask_size, mask_size, self.num_heads, depth, depth]
            ) + mask[:, None, :, :]
            attn = ops.reshape(attn, [-1, self.num_heads, depth, depth])

        attn = keras.activations.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)
        x = ops.matmul(attn, v)
        x = ops.transpose(x, [0, 2, 1, 3])
        x = ops.reshape(x, [batch_size, depth, channel])
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "window_size": self.window_size,
                "num_heads": self.num_heads,
                "qk_scale": self.qk_scale,
                "qkv_bias": self.qkv_bias,
                "attn_drop_rate": self.attn_drop_rate,
                "proj_drop_rate": self.proj_drop_rate,
            }
        )
        return config