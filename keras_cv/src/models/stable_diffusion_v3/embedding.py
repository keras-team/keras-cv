from keras_nlp.layers import RotaryEmbedding

from keras_cv.backend import keras


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