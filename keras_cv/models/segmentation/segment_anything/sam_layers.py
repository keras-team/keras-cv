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

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.layers.serializable_sequential import SerializableSequential


@keras_cv_export("keras_cv.layers.MLP")
class MLP(keras.layers.Layer):
    """A MLP block with architecture
    `input_dim -> [hidden_dim] * (num_layers - 1) -> output_dim`.

    Args:
        hidden_dim (int): The number of units in the hidden layers.
        output_dim (int): The number of units in the output layer.
        num_layers (int): The total number of dense layers to use.
        activation (str): Activation to use in the hidden layers.
            Default is `"relu"`.
    """

    def __init__(
        self, hidden_dim, output_dim, num_layers, activation="relu", **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation
        h = [hidden_dim] * (num_layers - 1)
        self.dense_net = []
        for hidden_dim in h:
            self.dense_net.append(keras.layers.Dense(hidden_dim))
            self.dense_net.append(keras.layers.Activation(activation))
        self.dense_net.append(keras.layers.Dense(output_dim))
        self.dense_net = SerializableSequential(self.dense_net)

    def build(self, input_shape):
        self.dense_net.build(input_shape)
        self.built = True

    def call(self, x):
        return self.dense_net(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
                "activation": self.activation,
            }
        )
        return config
