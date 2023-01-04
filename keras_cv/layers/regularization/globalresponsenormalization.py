# Copyright 2022 The KerasCV Authors
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


import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class GlobalResponseNormalization(layers.Layer):
    """
    Implementation of the GlobalResponseNormalization layer from
        [ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma",
            shape=(1, 1, 1, input_shape[-1]),
            initializer=tf.zeros_initializer(),
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(1, 1, 1, input_shape[-1]),
            initializer=tf.zeros_initializer(),
        )

    def call(self, inputs):
        Gx = tf.norm(inputs, ord=2, axis=(1, 2), keepdims=True)
        Nx = Gx / tf.reduce_mean(Gx, axis=-1, keepdims=True) + 1e-6
        return self.gamma * (inputs * Nx) + self.beta + inputs

    def get_config(self):
        config = {"gamma": self.gamma, "beta": self.beta}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
