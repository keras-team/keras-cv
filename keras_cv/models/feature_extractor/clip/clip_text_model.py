from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.models.feature_extractor.clip.clip_encoder import CLIPEncoder


class CLIPTextEncoder(keras.Model):
    def __init__(
        self,
        transformer_width,
        transformer_layers,
        transformer_heads,
        vocab_size,
        embed_dim,
        context_length,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.context_length = context_length
        self.token_embedding = keras.layers.Embedding(
            vocab_size,
            transformer_width,
            name="token_embedding",
        )

        self.vocab_size = vocab_size
        self.positional_embedding = self.add_weight(
            shape=[self.context_length, transformer_width],
            name="positional_embedding",
        )
        self.encoder = CLIPEncoder(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            name="residual_transformer_encoder",
        )
        self.ln_final = keras.layers.LayerNormalization(name="ln_final")

        self.text_projector = keras.layers.Dense(
            embed_dim, name="text_projector", use_bias=False
        )

    def call(self, inputs):
        token_embedding = self.token_embedding(inputs)
        encoded_output = self.encoder(
            token_embedding + self.positional_embedding
        )
        layer_norm = self.ln_final(encoded_output)
        indices = ops.expand_dims(
            ops.cast(ops.argmax(inputs, axis=1), "int32"), axis=-1
        )
        selected_features = ops.take_along_axis(
            layer_norm, indices[:, :, None], axis=1
        )
        text_features = self.text_projector(selected_features)
        output = ops.squeeze(text_features, axis=1)
        return output

    def build_attention_mask(self):
        mask = ops.ones((self.context_length, self.context_length))
        # Zero out the lower diagonal
        mask = ops.triu(mask)
        return ops.cast(mask, "float32")
