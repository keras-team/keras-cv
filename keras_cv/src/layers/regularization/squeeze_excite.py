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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops


@keras_cv_export("keras_cv.layers.SqueezeAndExcite2D")
class SqueezeAndExcite2D(keras.layers.Layer):
    """
    Implements Squeeze and Excite block as in
    [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf).
    This layer tries to use a content aware mechanism to assign channel-wise
    weights adaptively. It first squeezes the feature maps into a single value
    using global average pooling, which are then fed into two Conv1D layers,
    which act like fully-connected layers. The first layer reduces the
    dimensionality of the feature maps, and second layer restores it to its
    original value.

    The resultant values are the adaptive weights for each channel. These
    weights are then multiplied with the original inputs to scale the outputs
    based on their individual weightages.


    Args:
        filters: Number of input and output filters. The number of input and
            output filters is same.
        bottleneck_filters: (Optional) Number of bottleneck filters. Defaults
            to `0.25 * filters`
        squeeze_activation: (Optional) String, callable (or
            keras.layers.Layer) or keras.activations.Activation instance
            denoting activation to be applied after squeeze convolution.
            Defaults to `relu`.
        excite_activation: (Optional) String, callable (or
            keras.layers.Layer) or keras.activations.Activation instance
            denoting activation to be applied after excite convolution.
            Defaults to `sigmoid`.
    Example:

    ```python
    # (...)
    input = tf.ones((1, 5, 5, 16), dtype=tf.float32)
    x = keras.layers.Conv2D(16, (3, 3))(input)
    output = keras_cv.layers.SqueezeAndExciteBlock(16)(x)
    # (...)
    ```
    """

    def __init__(
        self,
        filters,
        bottleneck_filters=None,
        squeeze_activation="relu",
        excite_activation="sigmoid",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.filters = filters

        if bottleneck_filters and bottleneck_filters >= filters:
            raise ValueError(
                "`bottleneck_filters` should be smaller than `filters`. Got "
                f"`filters={filters}`, and "
                f"`bottleneck_filters={bottleneck_filters}`."
            )

        if filters <= 0 or not isinstance(filters, int):
            raise ValueError(
                f"`filters` should be a positive integer. Got {filters}"
            )

        self.bottleneck_filters = bottleneck_filters or (filters // 4)
        self.squeeze_activation = squeeze_activation
        self.excite_activation = excite_activation

        self.global_average_pool = keras.layers.GlobalAveragePooling2D(
            keepdims=True
        )
        self.squeeze_conv = keras.layers.Conv2D(
            self.bottleneck_filters,
            (1, 1),
            activation=self.squeeze_activation,
        )
        self.excite_conv = keras.layers.Conv2D(
            self.filters, (1, 1), activation=self.excite_activation
        )

    def call(self, inputs, training=None):
        x = self.global_average_pool(inputs)  # x: (batch_size, 1, 1, filters)
        x = self.squeeze_conv(x)  # x: (batch_size, 1, 1, bottleneck_filters)
        x = self.excite_conv(x)  # x: (batch_size, 1, 1, filters)
        x = ops.multiply(x, inputs)  # x: (batch_size, h, w, filters)
        return x

    def get_config(self):
        config = {
            "filters": self.filters,
            "bottleneck_filters": self.bottleneck_filters,
            "squeeze_activation": self.squeeze_activation,
            "excite_activation": self.excite_activation,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if isinstance(config["squeeze_activation"], dict):
            config["squeeze_activation"] = (
                keras.saving.deserialize_keras_object(
                    config["squeeze_activation"]
                )
            )
        if isinstance(config["excite_activation"], dict):
            config["excite_activation"] = keras.saving.deserialize_keras_object(
                config["excite_activation"]
            )
        return cls(**config)
