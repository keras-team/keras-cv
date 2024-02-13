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
        self.transformer_width = transformer_width
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.token_embedding = keras.layers.Embedding(
            vocab_size,
            transformer_width,
            name="token_embedding",
        )
        self.positional_embedding = keras.layers.Embedding(
            self.context_length,
            transformer_width,
            name="positional_embedding",
        )

        self.encoder = CLIPEncoder(
            width=transformer_width,
            num_layers=transformer_layers,
            heads=transformer_heads,
            name="clip_encoder",
        )
        self.ln_final = keras.layers.LayerNormalization(name="ln_final")

        self.text_projector = keras.layers.Dense(
            embed_dim, name="text_projector", use_bias=False
        )

    def call(self, inputs, attention_mask=None):
        token_embedding = self.token_embedding(inputs)
        position_ids = ops.expand_dims(
            ops.arange(self.context_length, dtype="int32"), 0
        )
        position_embedding = self.positional_embedding(position_ids)
        position_embedding = ops.tile(
            position_embedding, repeats=(inputs.shape[0], 1, 1)
        )
        causal_attention_mask = ops.ones(
            (self.context_length, self.context_length)
        )
        # Zero out the lower diagonal
        causal_attention_mask = ops.triu(causal_attention_mask)
        causal_attention_mask = ops.cast(causal_attention_mask, "float32")
        attention_mask = ops.cast(attention_mask, dtype="float32")
        expanded_mask = ops.tile(
            attention_mask[:, None, None, :], (1, 1, self.context_length, 1)
        )
        expanded_mask = (1.0 - expanded_mask) * (-1e8)
        encoded_output = self.encoder(
            token_embedding + position_embedding,
            causal_attention_mask=causal_attention_mask,
            attention_mask=expanded_mask,
        )
        layer_norm = self.ln_final(encoded_output)
        indices = ops.expand_dims(
            ops.cast(ops.argmax(inputs, axis=-1), "int32"), axis=-1
        )
        selected_features = ops.take_along_axis(
            layer_norm, indices[:, :, None], axis=1
        )
        text_features = self.text_projector(selected_features)
        output = ops.squeeze(text_features, axis=1)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "transformer_width": self.transformer_width,
                "transformer_layers": self.transformer_layers,
                "transformer_heads": self.transformer_heads,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "context_length": self.context_length,
            }
        )
        return config
