# Copyright 2023 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from keras_cv.src.backend import keras

try:
    import keras_nlp
    from keras_nlp.layers import RotaryEmbedding
except ImportError:
    keras_nlp = None
    StartEndPacker = None


class TimestepEmbedding(keras.layers.Layer):
    def __init__(self, hidden_dim, **kwargs):
        #
        # Config
        #
        self.hidden_dim = hidden_dim

        #
        # Layers
        #
        self.rotary_embedding = RotaryEmbedding()
        self.vector_embedding = VectorEmbedding(hidden_dim)

        #
        # Functional Model
        #
        timestep_input = keras.Input(
            shape=(None,), dtype="int32", name="timestep_input"
        )

        frequency = self.rotary_embedding(timestep_input)
        embedding = self.vector_embedding(frequency)

        super().__init__(
            inputs={"timestep_input": timestep_input},
            outputs={"embedding": embedding},
        )

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


class VectorEmbedding(keras.layers.Layer):
    def __init__(self, hidden_dim, **kwargs):
        #
        # Config
        #
        self.hidden_dim = hidden_dim

        #
        # Layers
        #
        self.dense1 = keras.layers.Dense(
            hidden_dim, activation=keras.activations.swish
        )
        self.dense2 = keras.layers.Dense(hidden_dim)

        #
        # Functional Model
        #
        vector_input = keras.Input(
            shape=(None,), dtype="int32", name="vector_input"
        )

        embedding = self.dense1(vector_input)
        embedding = self.dense2(embedding)

        super().__init__(
            inputs={"vector_input": vector_input},
            outputs={"embedding": embedding},
        )

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config
