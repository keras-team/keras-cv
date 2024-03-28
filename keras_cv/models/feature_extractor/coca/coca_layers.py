from keras import layers


class CoCaAttentionPooling(layers.Layer):
    """Implements the Pooled Attention Layer used in "coca": Contrastive Captioners are Image-Text Foundation Models"
    (https://arxiv.org/pdf/2205.01917.pdf), consisting of a Multiheaded Attention followed by Layer Normalization.

    Args:
        head_dim: The dimensions of the attention heads
        num_heads: The number of attention heads in the multi-headed attention layer
    """

    def __init__(self, head_dim, num_heads, **kwargs):
        super().__init__(**kwargs)

        self.head_dim = head_dim
        self.num_heads = num_heads

        self.multi_head_attn = layers.MultiHeadAttention(
            self.num_heads, self.head_dim
        )

        self.layer_norm = layers.LayerNormalization()

    def build(self, input_shape):
        # super().build(input_shape)

        if(len(input_shape) < 2):
            raise ValueError("Building CoCa Attention Pooling requires input shape of shape (query_shape, value_shape)")

        query_shape = input_shape[0]
        value_shape = input_shape[1]

        self.multi_head_attn._build_from_signature(query_shape, value_shape)
        self.layer_norm.build(query_shape)

    def call(self, query, value):
        x = self.multi_head_attn(query, value)
        return self.layer_norm(x)
