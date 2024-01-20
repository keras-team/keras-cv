import tensorflow as tf
from keras_nlp.layers import TransformerEncoder

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
            name="patch_embed.embedding",
        )
        self.width = width
        self.input_resolution = input_resolution
        self.patch_size = patch_size

    def build(self, input_shape):
        self.conv1.build(input_shape)
        self.class_embedding = self.add_weight(
            shape=((self.width,)), name="patch_embed.class_embedding"
        )

        self.positional_embedding = self.add_weight(
            shape=(
                (
                    (self.input_resolution // self.patch_size) ** 2 + 1,
                    self.width,
                )
            ),
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
    def __init__(self, width, layers, heads, attn_mask=None, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.layers = layers
        self.heads = heads
        self.attn_mask = attn_mask
        self.resblocks = keras.Sequential(
            [
                ResidualAttention(self.width, self.heads, self.attn_mask)
                for _ in range(self.layers)
            ]
        )

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        return self.resblocks(x)

    def compute_output_shape(self, inputs_shape):
        return inputs_shape


class ResidualAttention(keras.layers.Layer):
    def __init__(
        self,
        d_model,
        n_head,
        attn_mask=None,
    ):
        super().__init__()
        self.proj_dim = d_model
        self.n_head = n_head
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = (
            ops.cast(self.attn_mask, dtype=x.dtype)
            if self.attn_mask is not None
            else None
        )

        return self.attn(x, attention_mask=self.attn_mask)

    def build(self, input_shape):
        self.input_shape = input_shape
        super().build(input_shape)
        self.attn = CLIPAttention(
            self.proj_dim,
            self.n_head,
            name="multi_head_attention",
        )
        self.ln_1 = keras.layers.LayerNormalization(epsilon=1e-5, name="ln_1")
        self.mlp = keras.Sequential(
            [
                keras.layers.Dense(self.proj_dim * 4, name="c_fc"),
                QuickGELU(name="gelu"),
                keras.layers.Dense(self.proj_dim, name="c_proj"),
            ]
        )
        self.ln_2 = keras.layers.LayerNormalization(epsilon=1e-5, name="ln_2")

    def call(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def compute_output_shape(self, inputs_shape):
        return inputs_shape


class CLIPAttention(keras.layers.Layer):
    """
    - Documentation page: https://huggingface.co/docs/transformers/model_doc/clip
    - Implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py
    """

    def __init__(self, project_dim, num_heads, dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.project_dim = project_dim
        self.num_heads = num_heads
        self.head_dim = self.project_dim // self.num_heads
        if self.head_dim * self.num_heads != self.project_dim:
            raise ValueError(
                f"project_dim must be divisible by num_heads (got `project_dim`: {self.project_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.sqrt_att_head_size = ops.sqrt(self.head_dim)
        self.scale = self.head_dim**-0.5
        self.dropout = dropout
        self.q_proj = keras.layers.Dense(units=self.project_dim, name="q_proj")
        self.k_proj = keras.layers.Dense(units=self.project_dim, name="k_proj")
        self.v_proj = keras.layers.Dense(units=self.project_dim, name="v_proj")
        self.out_proj = keras.layers.Dense(
            units=self.project_dim, name="out_proj"
        )

    def build(self, input_shape):
        super().build(input_shape)
        self.q_proj.build(input_shape)
        self.k_proj.build(input_shape)
        self.v_proj.build(input_shape)
        self.out_proj.build(input_shape)

    def _transpose_for_scores(self, tensor, batch_size):
        """
        Copied from https://github.com/huggingface/transformers/blob/8e164c5400b7b413c7b8fb32e35132001effc970/src/transformers/models/bert/modeling_tf_bert.py#L252
        """
        # [batch_size, seq_len, all_head_dim] -> [batch_size, seq_len, num_heads, head_dim]
        tensor = ops.reshape(
            tensor, (batch_size, -1, self.num_heads, self.head_dim)
        )
        # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        return ops.transpose(tensor, axes=[0, 2, 1, 3])

    def call(
        self,
        x,
        attention_mask=None,
        causal_attention_mask=None,
        output_attentions=None,
        training=False,
    ):
        batch_size = ops.shape(x)[0]
        mixed_query_layer = self.q_proj(inputs=x)
        mixed_key_layer = self.k_proj(inputs=x)
        mixed_value_layer = self.v_proj(inputs=x)
        query_layer = self._transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self._transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self._transpose_for_scores(mixed_value_layer, batch_size)

        # Scaled dot product between key and query = raw attention scores.
        tf.print("transpose for scores shape", key_layer.shape)
        # attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = ops.matmul(
            query_layer, ops.transpose(key_layer, axes=[0, 1, 3, 2])
        )
        dk = ops.cast(ops.sqrt(self.head_dim), dtype=attention_scores.dtype)
        attention_scores = ops.divide(
            attention_scores, dk
        )  # (batch_size, num_heads, seq_len_q, seq_len_k)

        # Apply the causal_attention_mask first
        if causal_attention_mask is not None:
            # Apply the causal attention mask (precomputed for all layers in the call() function)
            attention_scores = ops.add(attention_scores, causal_attention_mask)

        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in the call() function)
            attention_scores = ops.add(attention_scores, attention_mask)

        # Normalize the attention scores to probabilities.
        _attention_probs = ops.softmax(attention_scores + 1e-9, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = keras.layers.Dropout(self.dropout)(
            inputs=_attention_probs, training=training
        )

        attn_output = ops.matmul(attention_probs, value_layer)
        attn_output = ops.transpose(attn_output, axes=[0, 2, 1, 3])

        # (batch_size, seq_len_q, project_dim)
        attn_output = ops.reshape(
            attn_output, (batch_size, -1, self.project_dim)
        )

        attn_output = self.out_proj(attn_output, training=training)
        outputs = (
            (attn_output, _attention_probs)
            if output_attentions
            else attn_output
        )

        return outputs
