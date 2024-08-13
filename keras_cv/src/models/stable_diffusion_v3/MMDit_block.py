# Copyright 2024 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writingf, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from keras_cv.src.backend import keras
from keras_cv.src.backend import ops


class MMDiTSelfAttention(keras.layers.Layer):
    def __init__(
        self, key_dim, num_heads, normalization_mode="rms_normalization"
    ):
        self.key_dim = key_dim
        self.head_dim = key_dim // num_heads
        self.normalization_mode = normalization_mode

        #
        # Layers
        #
        self.xdense = keras.layers.Dense(key_dim)
        self.cdense = keras.layers.Dense(key_dim)

        if normalization_mode == "rms_normalization":
            # TODO(varuns1997): Re-Implement RMSNormalization
            # for Keras 2 Compatibility
            self.query_normalization = keras.layers.LayerNormalization(
                rms_scaling=True
            )
            self.key_normalization = keras.layers.LayerNormalization(
                rms_scaling=True
            )
        elif normalization_mode == "layer_normalization":
            self.query_normalization = keras.layers.LayerNormalization()
            self.key_normalization = keras.layers.LayerNormalization()
        else:
            self.query_normalization = keras.layers.Identity()
            self.key_normalization = keras.layers.Identity()

        self.attn = keras.layers.MultiHeadAttention(num_heads, key_dim, key_dim)

        #
        # Functional Model
        #

        # Input Shape should be [batch_size, in_dim]
        context_in = keras.Input(
            shape=(None,), dtype="int32", name="context_in"
        )

        x_in = keras.Input(shape=(None,), dtype="int32", name="x_in")

        x = self.xdense(x_in)
        c = self.cdense(context_in)

        query_x = self.query_normalization(x)
        query_c = self.query_normalization(c)
        query = ops.concatenate([query_c, query_x], axis=1)

        key_x = self.key_normalization(x)
        key_c = self.key_normalization(c)
        key = ops.concatenate([key_c, key_x], axis=1)

        value = ops.concatenate([c, x], axis=1)

        attn_results = self.attn(query=query, value=value, key=key)

        c_attn = attn_results[:, : ops.shape(c)[1]]
        x_attn = attn_results[:, ops.shape(c)[1] :]

        super().__init__(
            inputs={
                "context_in": context_in,
                "x_in": x_in,
            },
            outputs={"c_attn": c_attn, "x_attn": x_attn},
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "key_dim": self.key_dim,
                "head_dim": self.head_dim,
                "normalization_mode": self.normalization_mode,
            }
        )
        return config


class AdaLNModulation(keras.layers.Layer):
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim

        #
        # Layers
        #
        self.silu = keras.activations.silu()
        self.dense = keras.layers.Dense(hidden_dim)

        #
        # Functional Model
        #
        x = keras.Input(shape=(None,), dtype="int32", name="y_in")

        intermediate = self.silu(x)
        y = self.dense(intermediate)

        super().__init__(
            inputs={
                "x": x,
            },
            outputs={
                "y": y,
            },
        )


def modulate(x, w, b):
    return x * (1 + ops.expand_dims(w, axis=1)) + ops.expand_dims(b, axis=1)


class MMDiTBlock(keras.layers.Layer):
    """A MMDiT block"""

    def __init__(
        self, key_dim, attn_heads, hidden_dim, normalization_mode="rms"
    ):
        self.key_dim = key_dim
        self.attn_heads = attn_heads
        self.normalization_mode = normalization_mode
        self.hidden_dim = hidden_dim

        #
        # Layers
        #
        self.cy_silu = AdaLNModulation(hidden_dim=hidden_dim)
        self.xy_silu = AdaLNModulation(hidden_dim=hidden_dim)

        self.layer_norm_c1 = keras.layers.LayerNormalization()
        self.layer_norm_x1 = keras.layers.LayerNormalization()

        self.self_attention = MMDiTSelfAttention(
            self.key_dim, self.attn_heads, self.normalization_mode
        )

        self.dense_c1 = keras.layers.Dense(self.hidden_dim)
        self.dense_x1 = keras.layers.Dense(self.hidden_dim)

        self.layer_norm_c2 = keras.layers.LayerNormalization()
        self.layer_norm_x2 = keras.layers.LayerNormalization()

        # MLPs of 2 dense layers each
        self.dense_c2 = keras.layers.Dense(self.hidden_dim, activation="relu")
        self.dense_c3 = keras.layers.Dense(self.hidden_dim, activation="relu")

        self.dense_x2 = keras.layers.Dense(self.hidden_dim, activation="relu")
        self.dense_x3 = keras.layers.Dense(self.hidden_dim, activation="relu")

        #
        # Functional Model
        #
        y_in = keras.Input(shape=(None,), dtype="int32", name="y_in")

        context_in = keras.Input(
            shape=(None,), dtype="int32", name="context_in"
        )

        x_in = keras.Input(shape=(None,), dtype="int32", name="x_in")

        yc = self.cy_silu(y_in)
        yx = self.xy_silu(y_in)

        (
            shift_msa_c,
            scale_msa_c,
            gate_msa_c,
            shift_mlp_c,
            scale_mlp_c,
            gate_mlp_c,
        ) = ops.split(yc, 6, axis=1)
        (
            shift_msa_x,
            scale_msa_x,
            gate_msa_x,
            shift_mlp_x,
            scale_mlp_x,
            gate_mlp_x,
        ) = ops.split(yx, 6, axis=1)

        c = self.layer_norm_c1(context_in)
        x = self.layer_norm_x1(x_in)

        c = modulate(c, shift_msa_c, scale_msa_c)
        x = modulate(x, shift_msa_x, scale_msa_x)

        c_attn, x_attn = self.self_attention(context_in=c, x_in=x)

        c = self.dense_c1(c_attn)
        x = self.dense_x1(c_attn)

        c = modulate(c, gate_msa_c, ops.zeros_like(gate_msa_c))
        x = modulate(x, gate_msa_x, ops.zeros_like(gate_msa_x))

        c = ops.add(c, context_in)
        x = ops.add(x, x_in)

        c = self.layer_norm_c2(c)
        x = self.layer_norm_x2(x)

        c = modulate(c, shift_mlp_c, ops.zeros_like(scale_mlp_c))
        x = modulate(x, shift_mlp_x, ops.zeros_like(scale_mlp_x))

        c = self.layer_norm_c3(c)
        x = self.layer_norm_x3(x)

        c = modulate(c, gate_mlp_c, ops.zeros_like(gate_mlp_c))
        x = modulate(x, gate_mlp_x, ops.zeros_like(gate_mlp_x))

        c_out = ops.add(c, context_in)
        x_out = ops.add(x, x_in)

        super().__init__(
            inputs={
                "y_in": y_in,
                "context_in": context_in,
                "x_in": x_in,
            },
            outputs={"c_out": c_out, "x_out": x_out},
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "key_dim": self.key_dim,
                "head_dim": self.head_dim,
                "normalization_mode": self.normalization_mode,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config
