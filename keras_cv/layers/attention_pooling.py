from keras import layers


class AttentionPooling(layers.Layer):
    """Implements the Pooled Attention Layer used in "CoCa": Contrastive Captioners are Image-Text Foundation Models"
    (https://arxiv.org/pdf/2205.01917.pdf), consisting of a Multiheaded Attention followed by Layer Normalization.

    Args:
        proj_dim: The dimensions of the attention heads
        num_heads: The number of attention heads in the multi-headed attention layer
    """
    def __init__(self,
                 proj_dim,
                 num_heads,
                 **kwargs):
        super().__init__(self, **kwargs)

        self.proj_dim = proj_dim
        self.num_heads = num_heads

        self.multi_head_attn = layers.MultiHeadAttention(
            self.num_heads,
            self.proj_dim
        )

        self.layer_norm = layers.LayerNormalization()

    def build(self, input_shape):
        super().build(input_shape)

        self.multi_head_attn.build(input_shape)
        self.layer_norm.build(input_shape)

    def call(self, query, value):
        x = self.multi_head_attn(query, value)
        return self.layer_norm(x)
