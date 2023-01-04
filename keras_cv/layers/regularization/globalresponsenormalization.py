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

        self.epsilon = 1e-6

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
        """
        Calls the GlobalResponseNormalization layer on the input.

        Original implementation's equivalent in TensorFlow is
        `Gx = tf.norm(inputs, ord=2, axis=(1, 2), keepdims=True)`
        But tf.norm() seems to cause TPU-related errors and slows down GPU training
        as pointed out by https://github.com/keras-team/keras-cv/pull/1234#issuecomment-1370554542
        """
        # Force float32 for numerical stability
        gamma = tf.cast(self.gamma, tf.float32)
        beta = tf.cast(self.beta, tf.float32)
        x = tf.cast(inputs, tf.float32)

        Gx = tf.pow(
            (
                tf.reduce_sum(tf.pow(inputs, 2), axis=(1, 2), keepdims=True)
                + self.epsilon
            ),
            0.5,
        )
        Nx = Gx / tf.reduce_mean(Gx, axis=-1, keepdims=True) + self.epsilon

        result = gamma * (x * Nx) + beta + inputs
        return tf.cast(result, inputs.dtype)
