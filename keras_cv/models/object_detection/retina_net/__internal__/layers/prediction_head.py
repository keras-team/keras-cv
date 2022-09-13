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
class PredictionHead(layers.Layer):
    """The class/box predictions head.

    Arguments:
      output_filters: Number of convolution filters in the final layer.
      bias_initializer: Bias Initializer for the final convolution layer.

    Returns:
      A function representing either the classification
        or the box regression head depending on `output_filters`.
    """

    def __init__(self, output_filters, bias_initializer, num_conv_layers=3, **kwargs):
        super().__init__(**kwargs)
        self.output_filters = output_filters
        self.bias_initializer = bias_initializer
        self.num_conv_layers = num_conv_layers

        self.conv_layers = [
            layers.Conv2D(
                256,
                kernel_size=3,
                padding="same",
                kernel_initializer=tf.keras.initializers.Orthogonal(),
                activation="relu",
            )
            for _ in range(num_conv_layers)
        ]
        self.prediction_layer = layers.Conv2D(
            self.output_filters,
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer=tf.keras.initializers.Orthogonal(),
            bias_initializer=self.bias_initializer,
        )

    def call(self, x, training=False):
        for layer in self.conv_layers:
            x = layer(x, training=training)
        x = self.prediction_layer(x, training=training)
        return x

    def get_config(self):
        config = {
            "bias_initializer": self.bias_initializer,
            "output_filters": self.output_filters,
            "num_conv_layers": self.num_conv_layers,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
