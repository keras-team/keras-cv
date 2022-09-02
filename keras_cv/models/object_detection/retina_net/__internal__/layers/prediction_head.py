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

from tensorflow.keras import initializers
from tensorflow.keras import layers


def PredictionHead(output_filters, bias_initializer):
    """The class/box predictions head.

    Arguments:
      output_filters: Number of convolution filters in the final layer.
      bias_initializer: Bias Initializer for the final convolution layer.

    Returns:
      A function representing either the classification
        or the box regression head depending on `output_filters`.
    """
    conv_layers = [
        layers.Conv2D(
            256,
            3,
            padding="same",
            kernel_initializer=initializers.RandomNormal(0.0, 0.01),
            activation="relu",
        )
        for _ in range(4)
    ]
    conv_layers += [
        layers.Conv2D(
            output_filters,
            3,
            1,
            padding="same",
            kernel_initializer=initializers.RandomNormal(0.0, 0.01),
            bias_initializer=bias_initializer,
        )
    ]

    def apply(x):
        for layer in conv_layers:
            x = layer(x)
        return x

    return apply
