import tensorflow as tf
from tensorflow import keras

from __internal__.layers.padded_conv2d import PaddedConv2D
from __internal__.layers.group_normalization import GroupNormalization


class DiffusionModel(keras.Model):
    def __init__(self, img_height, img_width, max_text_length, name=None):
        context = keras.layers.Input((max_text_length, 768))
        embed_input = keras.layers.Input((320,))
        latent = keras.layers.Input((img_height // 8, img_width // 8, 4))

        t_emb = keras.layers.Dense(1280)(embed_input)
        t_emb = keras.layers.Activation("swish")(t_emb)
        t_emb = keras.layers.Dense(1280)(t_emb)

        # Downsampling flow

        x = PaddedConv2D(320, kernel_size=3, padding=1)(latent)
        x1 = x

        x = ResBlock(320)([x, t_emb])
        x = SpatialTransformer(320, 8, 40)([x, context])
        x2 = x
        x = ResBlock(320)([x, t_emb])
        x = SpatialTransformer(320, 8, 40)([x, context])
        x3 = x
        x = Downsample(320)(x)
        x4 = x

        x = ResBlock(640)([x, t_emb])
        x = SpatialTransformer(640, 8, 80)([x, context])
        x5 = x
        x = ResBlock(640)([x, t_emb])
        x = SpatialTransformer(640, 8, 80)([x, context])
        x6 = x
        x = Downsample(640)(x)
        x7 = x

        x = ResBlock(1280)([x, t_emb])
        x = SpatialTransformer(1280, 8, 160)([x, context])
        x8 = x
        x = ResBlock(1280)([x, t_emb])
        x = SpatialTransformer(1280, 8, 160)([x, context])
        x9 = x
        x = Downsample(1280)(x)
        x10 = x

        x = ResBlock(1280)([x, t_emb])
        x11 = x
        x = ResBlock(1280)([x, t_emb])
        x12 = x

        # Middle flow

        x = ResBlock(1280)([x, t_emb])
        x = SpatialTransformer(1280, 8, 160)([x, context])
        x = ResBlock(1280)([x, t_emb])

        # Upsampling flow

        x = keras.layers.Concatenate()([x, x12])
        x = ResBlock(1280)([x, t_emb])

        x = keras.layers.Concatenate()([x, x11])
        x = ResBlock(1280)([x, t_emb])

        x = keras.layers.Concatenate()([x, x10])
        x = ResBlock(1280)([x, t_emb])
        x = Upsample(1280)(x)

        x = keras.layers.Concatenate()([x, x9])
        x = ResBlock(1280)([x, t_emb])
        x = SpatialTransformer(1280, 8, 160)([x, context])

        x = keras.layers.Concatenate()([x, x8])
        x = ResBlock(1280)([x, t_emb])
        x = SpatialTransformer(1280, 8, 160)([x, context])

        x = keras.layers.Concatenate()([x, x7])
        x = ResBlock(1280)([x, t_emb])
        x = SpatialTransformer(1280, 8, 160)([x, context])
        x = Upsample(1280)(x)

        x = keras.layers.Concatenate()([x, x6])
        x = ResBlock(640)([x, t_emb])
        x = SpatialTransformer(640, 8, 80)([x, context])

        x = keras.layers.Concatenate()([x, x5])
        x = ResBlock(640)([x, t_emb])
        x = SpatialTransformer(640, 8, 80)([x, context])

        x = keras.layers.Concatenate()([x, x4])
        x = ResBlock(640)([x, t_emb])
        x = SpatialTransformer(640, 8, 80)([x, context])
        x = Upsample(640)(x)

        x = keras.layers.Concatenate()([x, x3])
        x = ResBlock(320)([x, t_emb])
        x = SpatialTransformer(320, 8, 40)([x, context])

        x = keras.layers.Concatenate()([x, x2])
        x = ResBlock(320)([x, t_emb])
        x = SpatialTransformer(320, 8, 40)([x, context])

        x = keras.layers.Concatenate()([x, x1])
        x = ResBlock(320)([x, t_emb])
        x = SpatialTransformer(320, 8, 40)([x, context])

        # Exit flow

        x = GroupNormalization(epsilon=1e-5)(x)
        x = keras.layers.Activation("swish")(x)
        output = PaddedConv2D(4, kernel_size=3, padding=1)(x)

        super().__init__([latent, embed_input, context], output, name=name)


class ResBlock(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.in_layers = [
            GroupNormalization(epsilon=1e-5),
            keras.layers.Activation("swish"),
            PaddedConv2D(output_dim, 3, padding=1),
        ]
        self.emb_layers = [
            keras.layers.Activation("swish"),
            keras.layers.Dense(output_dim),
        ]
        self.out_layers = [
            GroupNormalization(epsilon=1e-5),
            keras.layers.Activation("swish"),
            PaddedConv2D(output_dim, 3, padding=1),
        ]

    def build(self, input_shape):
        if input_shape[0][-1] != self.output_dim:
            self.residual_projection = PaddedConv2D(self.output_dim, 1)
        else:
            self.residual_projection = lambda x: x

    def call(self, inputs):
        inputs, emb = inputs
        x = inputs
        for layer in self.in_layers:
            x = layer(x)
        y = emb
        for layer in self.emb_layers:
            y = layer(y)
        z = x + y[:, None, None]
        for layer in self.out_layers:
            z = layer(z)
        return z + self.residual_projection(inputs)


class SpatialTransformer(keras.layers.Layer):
    def __init__(self, channels, n_heads, d_head, **kwargs):
        super().__init__(**kwargs)
        self.norm = GroupNormalization(epsilon=1e-5)
        assert channels == n_heads * d_head
        self.proj_in = PaddedConv2D(n_heads * d_head, 1)
        self.transformer_blocks = [BasicTransformerBlock(channels, n_heads, d_head)]
        self.proj_out = PaddedConv2D(channels, 1)

    def call(self, inputs):
        inputs, context = inputs
        _, h, w, c = inputs.shape
        x = self.norm(inputs)
        x = self.proj_in(x)
        x = tf.reshape(x, (-1, h * w, c))
        for block in self.transformer_blocks:
            x = block([x, context])
        x = tf.reshape(x, (-1, h, w, c))
        return self.proj_out(x) + inputs


class CrossAttention(keras.layers.Layer):
    def __init__(self, n_heads, d_head, **kwargs):
        super().__init__(**kwargs)
        self.to_q = keras.layers.Dense(n_heads * d_head, use_bias=False)
        self.to_k = keras.layers.Dense(n_heads * d_head, use_bias=False)
        self.to_v = keras.layers.Dense(n_heads * d_head, use_bias=False)
        self.scale = d_head**-0.5
        self.num_heads = n_heads
        self.head_size = d_head
        self.to_out = [keras.layers.Dense(n_heads * d_head)]

    def call(self, inputs):
        assert type(inputs) is list
        if len(inputs) == 1:
            inputs = inputs + [None]
        x, context = inputs
        context = x if context is None else context
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        assert len(x.shape) == 3
        q = tf.reshape(q, (-1, x.shape[1], self.num_heads, self.head_size))
        k = tf.reshape(k, (-1, context.shape[1], self.num_heads, self.head_size))
        v = tf.reshape(v, (-1, context.shape[1], self.num_heads, self.head_size))

        q = tf.transpose(q, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)
        k = tf.transpose(k, (0, 2, 3, 1))  # (bs, num_heads, head_size, time)
        v = tf.transpose(v, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)

        score = td_dot(q, k) * self.scale
        weights = keras.activations.softmax(score)  # (bs, num_heads, time, time)
        attention = td_dot(weights, v)
        attention = tf.transpose(
            attention, (0, 2, 1, 3)
        )  # (bs, time, num_heads, head_size)
        x = tf.reshape(attention, (-1, x.shape[1], self.num_heads * self.head_size))
        for layer in self.to_out:
            x = layer(x)
        return x


class BasicTransformerBlock(keras.layers.Layer):
    def __init__(self, dim, n_heads, d_head, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn1 = CrossAttention(n_heads, d_head)

        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn2 = CrossAttention(n_heads, d_head)

        self.norm3 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.geglu = GEGLU(dim * 4)
        self.dense = keras.layers.Dense(dim)

    def call(self, inputs):
        inputs, context = inputs
        x = self.attn1([self.norm1(inputs)]) + inputs
        x = self.attn2([self.norm2(x), context]) + x
        return self.dense(self.geglu(self.norm3(x))) + x


class Downsample(keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.conv2d = PaddedConv2D(channels, 3, strides=2, padding=1)

    def call(self, x):
        return self.conv2d(x)


class Upsample(keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.ups = keras.layers.UpSampling2D(2)
        self.conv = PaddedConv2D(channels, 3, padding=1)

    def call(self, x):
        x = self.ups(x)
        return self.conv(x)


class GEGLU(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dense = keras.layers.Dense(output_dim * 2)

    def call(self, inputs):
        x = self.dense(inputs)
        x, gate = x[..., : self.output_dim], x[..., self.output_dim :]
        return x * gelu(gate)


def gelu(x):
    tanh_res = keras.activations.tanh(x * 0.7978845608 * (1 + 0.044715 * (x**2)))
    return 0.5 * x * (1 + tanh_res)


def td_dot(a, b):
    aa = tf.reshape(a, (-1, a.shape[2], a.shape[3]))
    bb = tf.reshape(b, (-1, b.shape[2], b.shape[3]))
    cc = keras.backend.batch_dot(aa, bb)
    return tf.reshape(cc, (-1, a.shape[1], cc.shape[1], cc.shape[2]))
