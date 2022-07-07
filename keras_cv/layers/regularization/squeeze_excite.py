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
class SqueezeAndExcite2D(layers.Layer):
    """
    Implements Squeeze and Excite block as in
    [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf).
    This layer tries to use a content aware mechanism to assign channel-wise
    weights adaptively. It first squeezes the feature maps into a single value
    using global average pooling, which are then fed into two Conv1D layers,
    which act like fully-connected layers. The first layer reduces the
    dimensionality of the feature maps by a factor of `ratio`, whereas the second
    layer restores it to its original value.

    The resultant values are the adaptive weights for each channel. These
    weights are then multiplied with the original inputs to scale the outputs
    based on their individual weightages.


    Args:
        filters: Number of input and output filters. The number of input and
            output filters is same.
        ratio: Ratio for bottleneck filters. Number of bottleneck filters =
            filters * ratio. Defaults to 0.25.
        squeeze_activation: (Optional) String, callable (or tf.keras.layers.Layer) or
            tf.keras.activations.Activation instance denoting activation to
            be applied after squeeze convolution. Defaults to `relu`.
        excite_activation: (Optional) String, callable (or tf.keras.layers.Layer) or
            tf.keras.activations.Activation instance denoting activation to
            be applied after excite convolution. Defaults to `sigmoid`.
    Usage:

    ```python
    # (...)
    input = tf.ones((1, 5, 5, 16), dtype=tf.float32)
    x = tf.keras.layers.Conv2D(16, (3, 3))(input)
    output = keras_cv.layers.SqueezeAndExciteBlock(16)(x)
    # (...)
    ```
    """

    def __init__(
        self,
        filters,
        ratio=0.25,
        squeeze_activation="relu",
        excite_activation="sigmoid",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.filters = filters

        if ratio <= 0.0 or ratio >= 1.0:
            raise ValueError(f"`ratio` should be a float between 0 and 1. Got {ratio}")

        if filters <= 0 or not isinstance(filters, int):
            raise ValueError(f"`filters` should be a positive integer. Got {filters}")

        self.ratio = ratio
        self.bottleneck_filters = int(self.filters * self.ratio)

        self.squeeze_activation = squeeze_activation
        self.excite_activation = excite_activation

        self.global_average_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.squeeze_conv = layers.Conv2D(
            self.bottleneck_filters,
            (1, 1),
            activation=self.squeeze_activation,
        )
        self.excite_conv = layers.Conv2D(
            self.filters, (1, 1), activation=self.excite_activation
        )

    def call(self, inputs, training=True):
        x = self.global_average_pool(inputs)  # x: (batch_size, 1, 1, filters)
        x = self.squeeze_conv(x)  # x: (batch_size, 1, 1, bottleneck_filters)
        x = self.excite_conv(x)  # x: (batch_size, 1, 1, filters)
        x = tf.math.multiply(x, inputs)  # x: (batch_size, h, w, filters)
        return x

    def get_config(self):
        config = {
            "filters": self.filters,
            "ratio": self.ratio,
            "squeeze_activation": self.squeeze_activation,
            "excite_activation": self.excite_activation,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
