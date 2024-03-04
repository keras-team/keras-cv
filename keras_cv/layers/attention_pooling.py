from keras import layers


class AttentionPooling(layers.Layer):

    # TODO: Add args
    def __init__(self,
                 proj_dim,
                 num_heads,
                 **kwargs):
        super().__init__(self, **kwargs)

        self.proj_dim = proj_dim
        self.num_heads = num_heads


    def build(self, input_shape):
        self.multi_head_attn = layers.MultiHeadAttention(
            self.num_heads,
            self.proj_dim
        )

        self.layer_norm = layers.LayerNormalization()

    def call(self, query, value, *args, **kwargs):
        x = self.multi_head_attn(query, value)
        return self.layer_norm(x)