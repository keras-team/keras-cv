# Copyright 2022 The KerasCV Authors
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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.models.stable_diffusion.padded_conv2d import PaddedConv2D


@keras_cv_export("keras_cv.models.stable_diffusion.DiffusionModel")
class DiffusionModel(keras.Model):
    def __init__(
        self,
        img_height,
        img_width,
        max_text_length,
        name=None,
        download_weights=True,
    ):
        context = keras.layers.Input((max_text_length, 768), name="context")
        t_embed_input = keras.layers.Input((320,), name="timestep_embedding")
        latent = keras.layers.Input(
            (img_height // 8, img_width // 8, 4), name="latent"
        )

        t_emb = keras.layers.Dense(1280)(t_embed_input)
        t_emb = keras.layers.Activation("swish")(t_emb)
        t_emb = keras.layers.Dense(1280)(t_emb)

        # Downsampling flow

        outputs = []
        x = PaddedConv2D(320, kernel_size=3, padding=1)(latent)
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(320)([x, t_emb])
            x = SpatialTransformer(8, 40, fully_connected=False)([x, context])
            outputs.append(x)
        x = PaddedConv2D(320, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(640)([x, t_emb])
            x = SpatialTransformer(8, 80, fully_connected=False)([x, context])
            outputs.append(x)
        x = PaddedConv2D(640, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            x = SpatialTransformer(8, 160, fully_connected=False)([x, context])
            outputs.append(x)
        x = PaddedConv2D(1280, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            outputs.append(x)

        # Middle flow

        x = ResBlock(1280)([x, t_emb])
        x = SpatialTransformer(8, 160, fully_connected=False)([x, context])
        x = ResBlock(1280)([x, t_emb])

        # Upsampling flow

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(1280)([x, t_emb])
        x = Upsample(1280)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(1280)([x, t_emb])
            x = SpatialTransformer(8, 160, fully_connected=False)([x, context])
        x = Upsample(1280)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(640)([x, t_emb])
            x = SpatialTransformer(8, 80, fully_connected=False)([x, context])
        x = Upsample(640)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(320)([x, t_emb])
            x = SpatialTransformer(8, 40, fully_connected=False)([x, context])

        # Exit flow

        x = keras.layers.GroupNormalization(epsilon=1e-5)(x)
        x = keras.layers.Activation("swish")(x)
        output = PaddedConv2D(4, kernel_size=3, padding=1)(x)

        super().__init__([latent, t_embed_input, context], output, name=name)

        if download_weights:
            diffusion_model_weights_fpath = keras.utils.get_file(
                origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_diffusion_model.h5",  # noqa: E501
                file_hash="8799ff9763de13d7f30a683d653018e114ed24a6a819667da4f5ee10f9e805fe",  # noqa: E501
            )
            self.load_weights(diffusion_model_weights_fpath)


class DiffusionModelV2(keras.Model):
    def __init__(
        self,
        img_height,
        img_width,
        max_text_length,
        name=None,
        download_weights=True,
    ):
        context = keras.layers.Input((max_text_length, 1024), name="context")
        t_embed_input = keras.layers.Input((320,), name="timestep_embedding")
        latent = keras.layers.Input(
            (img_height // 8, img_width // 8, 4), name="latent"
        )

        t_emb = keras.layers.Dense(1280)(t_embed_input)
        t_emb = keras.layers.Activation("swish")(t_emb)
        t_emb = keras.layers.Dense(1280)(t_emb)

        # Downsampling flow

        outputs = []
        x = PaddedConv2D(320, kernel_size=3, padding=1)(latent)
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(320)([x, t_emb])
            x = SpatialTransformer(5, 64, fully_connected=True)([x, context])
            outputs.append(x)
        x = PaddedConv2D(320, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(640)([x, t_emb])
            x = SpatialTransformer(10, 64, fully_connected=True)([x, context])
            outputs.append(x)
        x = PaddedConv2D(640, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            x = SpatialTransformer(20, 64, fully_connected=True)([x, context])
            outputs.append(x)
        x = PaddedConv2D(1280, 3, strides=2, padding=1)(x)  # Downsample 2x
        outputs.append(x)

        for _ in range(2):
            x = ResBlock(1280)([x, t_emb])
            outputs.append(x)

        # Middle flow

        x = ResBlock(1280)([x, t_emb])
        x = SpatialTransformer(20, 64, fully_connected=True)([x, context])
        x = ResBlock(1280)([x, t_emb])

        # Upsampling flow

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(1280)([x, t_emb])
        x = Upsample(1280)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(1280)([x, t_emb])
            x = SpatialTransformer(20, 64, fully_connected=True)([x, context])
        x = Upsample(1280)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(640)([x, t_emb])
            x = SpatialTransformer(10, 64, fully_connected=True)([x, context])
        x = Upsample(640)(x)

        for _ in range(3):
            x = keras.layers.Concatenate()([x, outputs.pop()])
            x = ResBlock(320)([x, t_emb])
            x = SpatialTransformer(5, 64, fully_connected=True)([x, context])

        # Exit flow

        x = keras.layers.GroupNormalization(epsilon=1e-5)(x)
        x = keras.layers.Activation("swish")(x)
        output = PaddedConv2D(4, kernel_size=3, padding=1)(x)

        super().__init__([latent, t_embed_input, context], output, name=name)

        if download_weights:
            diffusion_model_weights_fpath = keras.utils.get_file(
                origin="https://huggingface.co/ianstenbit/keras-sd2.1/resolve/main/diffusion_model_v2_1.h5",  # noqa: E501
                file_hash="c31730e91111f98fe0e2dbde4475d381b5287ebb9672b1821796146a25c5132d",  # noqa: E501
            )
            self.load_weights(diffusion_model_weights_fpath)


class ResBlock(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.entry_flow = [
            keras.layers.GroupNormalization(epsilon=1e-5),
            keras.layers.Activation("swish"),
            PaddedConv2D(output_dim, 3, padding=1),
        ]
        self.embedding_flow = [
            keras.layers.Activation("swish"),
            keras.layers.Dense(output_dim),
        ]
        self.exit_flow = [
            keras.layers.GroupNormalization(epsilon=1e-5),
            keras.layers.Activation("swish"),
            PaddedConv2D(output_dim, 3, padding=1),
        ]

    def build(self, input_shape):
        if input_shape[0][-1] != self.output_dim:
            self.residual_projection = PaddedConv2D(self.output_dim, 1)
        else:
            self.residual_projection = lambda x: x

    def call(self, inputs):
        inputs, embeddings = inputs
        x = inputs
        for layer in self.entry_flow:
            x = layer(x)
        for layer in self.embedding_flow:
            embeddings = layer(embeddings)
        x = x + embeddings[:, None, None]
        for layer in self.exit_flow:
            x = layer(x)
        return x + self.residual_projection(inputs)


class SpatialTransformer(keras.layers.Layer):
    def __init__(self, num_heads, head_size, fully_connected=False, **kwargs):
        super().__init__(**kwargs)
        self.norm = keras.layers.GroupNormalization(epsilon=1e-5)
        channels = num_heads * head_size
        if fully_connected:
            self.proj1 = keras.layers.Dense(num_heads * head_size)
        else:
            self.proj1 = PaddedConv2D(num_heads * head_size, 1)
        self.transformer_block = BasicTransformerBlock(
            channels, num_heads, head_size
        )
        if fully_connected:
            self.proj2 = keras.layers.Dense(channels)
        else:
            self.proj2 = PaddedConv2D(channels, 1)

    def call(self, inputs):
        inputs, context = inputs
        _, h, w, c = inputs.shape
        x = self.norm(inputs)
        x = self.proj1(x)
        x = ops.reshape(x, (-1, h * w, c))
        x = self.transformer_block([x, context])
        x = ops.reshape(x, (-1, h, w, c))
        return self.proj2(x) + inputs


class BasicTransformerBlock(keras.layers.Layer):
    def __init__(self, dim, num_heads, head_size, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn1 = CrossAttention(num_heads, head_size)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn2 = CrossAttention(num_heads, head_size)
        self.norm3 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.geglu = GEGLU(dim * 4)
        self.dense = keras.layers.Dense(dim)

    def call(self, inputs):
        inputs, context = inputs
        x = self.attn1(self.norm1(inputs), context=None) + inputs
        x = self.attn2(self.norm2(x), context=context) + x
        return self.dense(self.geglu(self.norm3(x))) + x


class CrossAttention(keras.layers.Layer):
    def __init__(self, num_heads, head_size, **kwargs):
        super().__init__(**kwargs)
        self.to_q = keras.layers.Dense(num_heads * head_size, use_bias=False)
        self.to_k = keras.layers.Dense(num_heads * head_size, use_bias=False)
        self.to_v = keras.layers.Dense(num_heads * head_size, use_bias=False)
        self.scale = head_size**-0.5
        self.num_heads = num_heads
        self.head_size = head_size
        self.out_proj = keras.layers.Dense(num_heads * head_size)

    def call(self, inputs, context=None):
        if context is None:
            context = inputs
        q, k, v = self.to_q(inputs), self.to_k(context), self.to_v(context)
        q = ops.reshape(
            q, (-1, inputs.shape[1], self.num_heads, self.head_size)
        )
        k = ops.reshape(
            k, (-1, context.shape[1], self.num_heads, self.head_size)
        )
        v = ops.reshape(
            v, (-1, context.shape[1], self.num_heads, self.head_size)
        )

        q = ops.transpose(q, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)
        k = ops.transpose(k, (0, 2, 3, 1))  # (bs, num_heads, head_size, time)
        v = ops.transpose(v, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)

        score = td_dot(q, k) * self.scale
        weights = keras.activations.softmax(
            score
        )  # (bs, num_heads, time, time)
        attn = td_dot(weights, v)
        attn = ops.transpose(
            attn, (0, 2, 1, 3)
        )  # (bs, time, num_heads, head_size)
        out = ops.reshape(
            attn, (-1, inputs.shape[1], self.num_heads * self.head_size)
        )
        return self.out_proj(out)


class Upsample(keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.ups = keras.layers.UpSampling2D(2)
        self.conv = PaddedConv2D(channels, 3, padding=1)

    def call(self, inputs):
        return self.conv(self.ups(inputs))


class GEGLU(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dense = keras.layers.Dense(output_dim * 2)

    def call(self, inputs):
        x = self.dense(inputs)
        x, gate = x[..., : self.output_dim], x[..., self.output_dim :]
        tanh_res = keras.activations.tanh(
            gate * 0.7978845608 * (1 + 0.044715 * (gate**2))
        )
        return x * 0.5 * gate * (1 + tanh_res)


def td_dot(a, b):
    aa = ops.reshape(a, (-1, a.shape[2], a.shape[3]))
    bb = ops.reshape(b, (-1, b.shape[2], b.shape[3]))
    cc = keras.layers.Dot(axes=(2, 1))([aa, bb])
    return ops.reshape(cc, (-1, a.shape[1], cc.shape[1], cc.shape[2]))
