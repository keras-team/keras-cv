from keras_cv.backend import keras
from keras_cv.layers.vit_layers import PatchingAndEmbedding
from keras_cv.models.stable_diffusion.v3 import embedding
from keras_cv.models.stable_diffusion.v3.MMDiT_block import MMDiTBlock


class MMDiT(keras.layers.Layer):
    """Multimodal Transformer Diffusion model."""

    def __init__(
        self,
        num_blocks,
        image_embed_dim,
        patch_size,
        timestep_embed_dim,
        attn_heads,
        hidden_dim,
        normalization_mode="rms_normalization",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_blocks = num_blocks
        self.image_embed_dim = image_embed_dim
        self.patch_size = patch_size
        self.timestep_embed_dim = timestep_embed_dim
        self.attn_heads = attn_heads
        self.hidden_dim = hidden_dim
        self.normalization_mode = normalization_mode

        #
        # Layers
        #
        self.image_patching = PatchingAndEmbedding(image_embed_dim, patch_size)
        self.timestep_embedding = embedding.TimestepEmbedding(
            timestep_embed_dim, hidden_dim
        )

        self.blocks = [
            MMDiTBlock(
                hidden_dim, attn_heads, hidden_dim, qk_norm=normalization_mode
            )
            for i in range(num_blocks)
        ]

        self.dense = keras.layers.Dense(hidden_dim)
        self.unpatching = None

        #
        # Functional Model
        #
        noised_latent = keras.Input(
            shape=(None,), dtype="int32", name="noised_latent"
        )

        x = self.image_patching(noised_latent)

        # TODO: Build C and Y from the text encoders
        c = None
        y = None

        for block in self.blocks:
            c, x = block(context_in=c, y_in=y, x_in=x)

        # TODO: Clarify Modulation (paper vs implementation discrepancy)

        x = self.dense(x)

        # TODO: Unpatching

        x_out = None

        super().__init__(
            inputs={"noised_latent": noised_latent}, outputs={"x_out": x_out}
        )
