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
from tensorflow.python.keras import layers


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class CBAM(layers.Layer):
    """
        Implements Convolution Block Attention Module (CBAM) as in
        [Convolution-Block-Attention--Module](https://arxiv.org/pdf/1807.06521.pdf).

        Args:
            filters: Number of input and output filters. The number of input and
                output filters is same.
            ratio: Ratio for bottleneck filters. Number of bottleneck filters =
                filters * ratio. Defaults to 0.25.
            channel_activation: (Optional) String, callable (or tf.keras.layers.Layer) or
                tf.keras.activations.Activation instance denoting activation to
                be applied after squeeze convolution. Defaults to `relu`.
            spatial_activation: (Optional) String, callable (or tf.keras.layers.Layer) or
                tf.keras.activations.Activation instance denoting activation to
                be applied after excite convolution. Defaults to `sigmoid`.
        Usage:
        ```python
        # (...)
        input = tf.ones((1, 5, 5, 16), dtype=tf.float32)
        x = tf.keras.layers.Conv2D(16, (3, 3))(input)
        output = keras_cv.layers.CBAM(16)(x)
        # (...)
        ```
    """
    def __init__(
        self,
        filters,
        ratio=0.25,
        channel_activation="relu",
        spatial_activation="sigmoid",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if ratio <= 0.0 or ratio >= 1.0:
            raise ValueError(f"`ratio` should be a float between 0 and 1. Got {ratio}")

        if filters <= 0 or not isinstance(filters, int):
            raise ValueError(f"`filters` should be a positive integer. Got {filters}")

        self.filters = filters
        self.ratio = ratio
        self.bottleneck_filters = int(self.filters * self.ratio)
        self.activation = layers.Activation("sigmoid")
        self.channel_activation = channel_activation
        self.spatial_activation = spatial_activation

        # Channels attention module
        self.channel_average_layer = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(keepdims=True),
            layers.Dense(self.bottleneck_filters, activation=self.channel_activation, use_bias=False),
            layers.Dense(self.filters, use_bias=False)
        ])
        self.channel_max_layer = tf.keras.Sequential([
            layers.GlobalMaxPooling2D(keepdims=True),
            layers.Dense(self.bottleneck_filters, activation=self.channel_activation, use_bias=False),
            layers.Dense(self.filters, use_bias=False)
        ])

        # Spatial attention module
        self.concat = layers.Concatenate()
        self.spatial_layer = layers.Conv2D(filters=1, kernel_size=7, padding="same", activation=self.spatial_activation)

    # inputs: (B, H, W, C)
    def call(self, inputs, training=True):
        channel_x1 = self.channel_average_layer(inputs)                # channels_x1: (B, C)
        channel_x2 = self.channel_max_layer(inputs)                    # channels_x2: (B, C)
        channel_activation = self.activation(channel_x1 + channel_x2)  # channel_activation:  (B, C)
        channel_attention = inputs * channel_activation                # channel_attention:  (B, H, W, C)

        spatial_x1 = tf.math.reduce_mean(channel_attention, axis=-1, keepdims=True)     # spatial_x1: (B, H, W, 1)
        spatial_x2 = tf.math.reduce_max(channel_attention, axis=-1, keepdims=True)      # spatial_x2: (B, H, W, 1)
        feats = self.concat([spatial_x1, spatial_x2])                                   # spatial_x2: (B, H, W, 2)
        spatial_conv = self.spatial_layer(feats)                                        # spatial_conv: (B, H, W, 1)
        outputs = channel_attention * spatial_conv                                      # outputs: (B, H, W, C)
        return outputs

    def get_config(self):
        config = {
            "filters": self.filters,
            "ratio": self.ratio,
            "channel_activation": self.channel_activation,
            "spatial_activation": self.spatial_activation,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
