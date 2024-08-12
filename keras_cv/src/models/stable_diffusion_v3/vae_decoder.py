# Copyright 2024 The KerasCV Authors
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


# =============== VAE Decoder for Stable Diffusion V3 ================

# Adapted from KerasCV's Stable Diffusion V1 and V2 models
# https://github.com/keras-team/keras-cv/tree/v0.8.2/keras_cv/models/stable_diffusion  # noqa: E501
# Also see: https://github.com/divamgupta/stable-diffusion-tensorflow/tree/master/stable_diffusion_tf  # noqa: E501

from keras_cv.src.backend import keras
from keras_cv.src.backend import ops


class VAEDecoder(keras.Model):
    def __init__(
        self,
        input_shape=(None, None, 16),
        hidden_channels=(512, 512, 256, 128),
        output_channels=3,
        num_res_blocks=2,
        **kwargs,
    ):
        decoder_layers = [
            PaddedConv2D(
                hidden_channels[0],
                kernel_size=3,
                strides=1,
                padding=1,
                name="input_projection",
            ),
            ResnetBlock(hidden_channels[0], name="input_block_1"),
            AttentionBlock(hidden_channels[0], name="input_attention"),
            ResnetBlock(hidden_channels[0], name="input_block_2"),
        ]
        for i, channels in enumerate(hidden_channels):
            for j in range(num_res_blocks + 1):
                decoder_layers.append(
                    ResnetBlock(channels, name=f"block_{i + 1}_{j + 1}")
                )
            if i != len(hidden_channels) - 1:  # no upsampling in the last block
                decoder_layers.append(
                    Upsample(hidden_channels[i], name=f"upsample_{i + 1}")
                )
        decoder_layers.extend(
            [
                keras.layers.GroupNormalization(
                    groups=32, epsilon=1e-6, name="output_norm"
                ),
                keras.layers.Activation("swish", name="output_activation"),
                PaddedConv2D(
                    output_channels,
                    kernel_size=3,
                    strides=1,
                    padding=1,
                    name="output_projection",
                ),
            ]
        )

        self.decoder = keras.Sequential(decoder_layers, name="decoder")

        latent = keras.Input(shape=input_shape, name="latent")
        upsampled_image = self.decoder(latent)

        super().__init__(inputs=latent, outputs=upsampled_image, **kwargs)

        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.num_res_blocks = num_res_blocks

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape[1:],
                "hidden_channels": self.hidden_channels,
                "num_res_blocks": self.num_res_blocks,
            }
        )
        return config


class ResnetBlock(keras.layers.Layer):
    def __init__(self, output_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        output_dim = (
            self.output_dim if self.output_dim is not None else input_shape[-1]
        )

        prev_output_shape = input_shape

        self.norm1 = keras.layers.GroupNormalization(groups=32, epsilon=1e-6)
        self.norm1.build(prev_output_shape)
        self.conv1 = PaddedConv2D(output_dim, kernel_size=3, padding=1)
        self.conv1.build(prev_output_shape)

        prev_output_shape = self.conv1.compute_output_shape(prev_output_shape)

        self.norm2 = keras.layers.GroupNormalization(groups=32, epsilon=1e-6)
        self.norm2.build(prev_output_shape)
        self.conv2 = PaddedConv2D(output_dim, kernel_size=3, padding=1)
        self.conv2.build(prev_output_shape)

        if input_shape[-1] != output_dim:
            self.residual_projection = PaddedConv2D(output_dim, kernel_size=1)
            self.residual_projection.build(input_shape)
        else:
            self.residual_projection = lambda x: x

        self.built = True

    def call(self, inputs):
        x = self.conv1(keras.activations.swish(self.norm1(inputs)))
        x = self.conv2(keras.activations.swish(self.norm2(x)))
        return x + self.residual_projection(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"output_dim": self.output_dim})
        return config


class AttentionBlock(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.group_norm = keras.layers.GroupNormalization(
            groups=32, epsilon=1e-6
        )
        self.query_proj = PaddedConv2D(output_dim, kernel_size=1)
        self.key_proj = PaddedConv2D(output_dim, kernel_size=1)
        self.value_proj = PaddedConv2D(output_dim, kernel_size=1)
        self.out_proj = PaddedConv2D(output_dim, kernel_size=1)

    def call(self, inputs):
        x = self.group_norm(inputs)
        q, k, v = self.query_proj(x), self.key_proj(x), self.value_proj(x)

        # Compute attention
        shape = ops.shape(q)
        h, w, c = shape[1], shape[2], shape[3]
        q = ops.reshape(q, (-1, h * w, c))  # b, hw, c
        k = ops.transpose(k, (0, 3, 1, 2))
        k = ops.reshape(k, (-1, c, h * w))  # b, c, hw
        y = q @ k
        y = y * 1 / ops.sqrt(ops.cast(c, self.compute_dtype))
        y = keras.activations.softmax(y)

        # Attend to values
        v = ops.transpose(v, (0, 3, 1, 2))
        v = ops.reshape(v, (-1, c, h * w))
        y = ops.transpose(y, (0, 2, 1))
        x = v @ y
        x = ops.transpose(x, (0, 2, 1))
        x = ops.reshape(x, (-1, h, w, c))

        return self.out_proj(x) + inputs

    def get_config(self):
        config = super().get_config()
        config.update({"output_dim": self.output_dim})
        return config


class Upsample(keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.upsample_layer = keras.layers.UpSampling2D(
            2, name="upsample_layer"
        )
        self.conv_layer = PaddedConv2D(
            num_channels, kernel_size=3, strides=1, padding=1, name="conv_layer"
        )

    def call(self, inputs):
        return self.conv_layer(self.upsample_layer(inputs))

    def get_config(self):
        config = super().get_config()
        config.update({"num_channels": self.num_channels})
        return config


class PaddedConv2D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding=0, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides

    def build(self, input_shape):
        self.padding2d = keras.layers.ZeroPadding2D(self.padding)
        self.padding2d.build(input_shape)
        input_shape = self.padding2d.compute_output_shape(input_shape)
        self.conv2d = keras.layers.Conv2D(
            self.filters, self.kernel_size, strides=self.strides
        )
        self.conv2d.build(input_shape)
        self.built = True

    def compute_output_shape(self, input_shape):
        input_shape = self.padding2d.compute_output_shape(input_shape)
        return self.conv2d.compute_output_shape(input_shape)

    def call(self, inputs):
        x = self.padding2d(inputs)
        return self.conv2d(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "padding": self.padding,
                "strides": self.strides,
            }
        )
        return config
