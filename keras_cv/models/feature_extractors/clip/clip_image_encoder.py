from keras_cv.backend import keras
from keras_cv.backend import ops


class CLIPPatchingAndEmbedding(keras.layers.Layer):
    def __init__(self, width, patch_size, input_resolution):
        super().__init__()

        self.conv1 = keras.layers.Conv2D(
            filters=width,
            kernel_size=patch_size,
            strides=patch_size,
            use_bias=False,
        )

        self.class_embedding = self.add_weight(
            shape=((width,)), name="patch_embed.class_embedding", trainable=True
        )

        self.positional_embedding = self.add_weight(
            shape=(((input_resolution // patch_size) ** 2 + 1, width)),
            trainable=True,
            name="patch_embed.positional_embedding",
        )

    def call(self, x):
        x = self.conv1(x)  # shape = [*, grid, grid, width]
        x = ops.transpose(
            x, axes=[0, 3, 1, 2]
        )  # shape = [*, width, grid, grid]
        shape = ops.shape(x)
        x = ops.reshape(
            x, [shape[0], shape[1], shape[2] * shape[3]]
        )  # shape = [*, width, grid ** 2]
        x = ops.transpose(x, axes=(0, 2, 1))  # shape = [*, grid ** 2, width]

        class_embedding = self.class_embedding

        shape = ops.shape(x)
        class_embedding_expanded = ops.expand_dims(class_embedding, axis=0)
        class_embedding_expanded = ops.expand_dims(
            class_embedding_expanded, axis=1
        )
        class_embedding_expanded = ops.tile(
            class_embedding_expanded, (shape[0], 1, 1)
        )
        x = ops.concatenate(
            [class_embedding_expanded, x], axis=1
        )  # shape = [*, grid ** 2 + 1, width]
        positional_embedding = self.positional_embedding
        x = x + positional_embedding

        return x


class QuickGELU(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return x * ops.sigmoid(1.702 * x)


class ResidualTransformerEncoder(keras.layers.Layer):
    def __init__(self, width, layers, heads, attn_mask=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = keras.Sequential(
            [ResidualAttention(width, heads, attn_mask) for _ in range(layers)]
        )

    def call(self, x):
        return self.resblocks(x)


class ResidualAttention(keras.layers.Layer):
    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()

        self.attn = keras.layers.MultiHeadAttention(n_head, d_model)
        self.ln_1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.mlp = keras.Sequential(
            [
                keras.layers.Dense(d_model * 4, name="c_fc"),
                QuickGELU(name="gelu"),
                keras.layers.Dense(d_model, name="c_proj"),
            ]
        )
        self.ln_2 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn_mask = attn_mask
        self.q_proj = keras.layers.Dense(units=d_model, name="q_proj")
        self.k_proj = keras.layers.Dense(units=d_model, name="k_proj")
        self.v_proj = keras.layers.Dense(units=d_model, name="v_proj")

    def attention(self, x):
        self.attn_mask = (
            ops.cast(self.attn_mask, dtype=x.dtype)
            if self.attn_mask is not None
            else None
        )

        key = self.k_proj(inputs=x)
        value = self.v_proj(inputs=x)
        query = self.q_proj(inputs=x)
        return self.attn(
            key=key, value=value, query=query, attention_mask=self.attn_mask
        )

    def call(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CLIPImageEncoder(keras.Model):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        input_tensor=None,
        **kwargs,
    ):
        inputs = keras.layers.Input(
            tensor=input_tensor, shape=(input_resolution, input_resolution, 3)
        )
        x = inputs

        x = CLIPPatchingAndEmbedding(
            width=width,
            patch_size=patch_size,
            input_resolution=input_resolution,
        )(x)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

        x = ops.transpose(x, axes=(1, 0, 2))
        x = ResidualTransformerEncoder(width, layers, heads)(x)
        x = ops.transpose(x, axes=(1, 0, 2))

        x = keras.layers.LayerNormalization(epsilon=1e-6)(x[:, 0, :])

        proj = keras.layers.Dense(output_dim)
        x = proj(x)

        output = x

        super().__init__(
            inputs=inputs,
            outputs=output,
            **kwargs,
        )
