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
from typing import Any
from typing import List
from typing import Mapping

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class SpatialPyramidPooling(tf.keras.layers.Layer):
    """Implements the Atrous Spatial Pyramid Pooling.

    References:
        [Rethinking Atrous Convolution for Semantic Image Segmentation](
          https://arxiv.org/pdf/1706.05587.pdf)
    [Encoder-Decoder with Atrous Separable Convolution for Semantic Image
        Segmentation](https://arxiv.org/pdf/1802.02611.pdf)

    inp = tf.keras.layers.Input((384, 384, 3))
    backbone = tf.keras.applications.EfficientNetB0(input_tensor=inp, include_top=False)
    output = backbone(inp)
    output = keras_cv.layers.SpatialPyramidPooling(
        dilation_rates=[6, 12, 18])(output)

    # output[4].shape = [None, 16, 16, 256]
    """

    def __init__(
        self,
        dilation_rates: List[int],
        num_channels: int = 256,
        activation: str = "relu",
        dropout: float = 0.0,
        **kwargs,
    ):
        """Initializes an Atrous Spatial Pyramid Pooling layer.

        Args:
            dilation_rates: A `list` of integers for parallel dilated conv. Usually a
                sample choice of rates are [6, 12, 18].
            num_channels: An `int` number of output channels. Default to 256.
            activation: A `str` activation to be used. Default to 'relu'.
            dropout: A `float` for the dropout rate of the final projection output after
                the activations and batch norm. Default to 0.0, which means no dropout is
                applied to the output.
            **kwargs: Additional keyword arguments to be passed.
        """
        super().__init__(**kwargs)
        self.dilation_rates = dilation_rates
        self.num_channels = num_channels
        self.activation = activation
        self.dropout = dropout

    def build(self, input_shape):

        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]

        # This is the parallel networks that process the input features with different
        # dilation rates. The output from each channel will be merged together and feed
        # to the output.
        self.aspp_parallel_channels = []

        # Channel1 with Conv2D and 1x1 kernel size.
        conv_sequential = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=self.num_channels,
                    kernel_size=(1, 1),
                    use_bias=False,
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(self.activation),
            ]
        )
        self.aspp_parallel_channels.append(conv_sequential)

        # Channel 2 and afterwards are based on self.dilation_rates, and each of them
        # will have conv2D with 3x3 kernel size.
        for dilation_rate in self.dilation_rates:
            conv_sequential = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(
                        filters=self.num_channels,
                        kernel_size=(3, 3),
                        padding="same",
                        dilation_rate=dilation_rate,
                        use_bias=False,
                    ),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation(self.activation),
                ]
            )
            self.aspp_parallel_channels.append(conv_sequential)

        # Last channel is the global average pooling with conv2D 1x1 kernel.
        pool_sequential = tf.keras.Sequential(
            [
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Reshape((1, 1, channels)),
                tf.keras.layers.Conv2D(
                    filters=self.num_channels,
                    kernel_size=(1, 1),
                    use_bias=False,
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(self.activation),
                tf.keras.layers.Resizing(height, width, interpolation="bilinear"),
            ]
        )
        self.aspp_parallel_channels.append(pool_sequential)

        # Final projection layers
        self.projection = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=self.num_channels,
                    kernel_size=(1, 1),
                    use_bias=False,
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(self.activation),
                tf.keras.layers.Dropout(rate=self.dropout),
            ],
        )

    def call(self, inputs, training=None):
        """Calls the Atrous Spatial Pyramid Pooling layer on an input.

        Args:
          inputs: A `tf.Tensor` of shape [batch, height, width, channels]

        Returns:
          A `tf.Tensor` of shape [batch, height, width, num_channels]
        """
        result = []

        for channel in self.aspp_parallel_channels:
            temp = tf.cast(channel(inputs, training=training), inputs.dtype)
            result.append(temp)

        result = tf.concat(result, axis=-1)
        result = self.projection(result, training=training)
        return result

    def get_config(self) -> Mapping[str, Any]:
        config = {
            "dilation_rates": self.dilation_rates,
            "num_channels": self.num_channels,
            "activation": self.activation,
            "dropout": self.dropout,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
