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
from keras import backend
from tensorflow.keras import layers

BN_AXIS = 3

CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 2.0,
        "mode": "fan_out",
        "distribution": "truncated_normal",
    },
}


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class MBConvBlock(layers.Layer):
    def __init__(
        self,
        input_filters: int,
        output_filters: int,
        expand_ratio=1,
        kernel_size=3,
        strides=1,
        se_ratio=0.0,
        bn_momentum=0.9,
        activation="swish",
        survival_probability: float = 0.8,
        name=None,
    ):
        """MBConv block: Mobile Inverted Residual Bottleneck."""

        self.input_filters = input_filters
        self.output_filters = output_filters
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.strides = strides
        self.se_ratio = se_ratio
        self.bn_momentum = bn_momentum
        self.activation = activation
        self.survival_probability = survival_probability
        self.name = name

    def build(self):
        if self.name is None:
            self.name = backend.get_uid("block0")

    def call(self, inputs):
        # Expansion phase
        filters = self.input_filters * self.expand_ratio
        if self.expand_ratio != 1:
            x = layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=1,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                padding="same",
                data_format="channels_last",
                use_bias=False,
                name=self.name + "expand_conv",
            )(inputs)
            x = layers.BatchNormalization(
                axis=BN_AXIS,
                momentum=self.bn_momentum,
                name=self.name + "expand_bn",
            )(x)
            x = layers.Activation(
                self.activation, name=self.name + "expand_activation"
            )(x)
        else:
            x = inputs

        # Depthwise conv
        x = layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=self.strides,
            depthwise_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            data_format="channels_last",
            use_bias=False,
            name=self.name + "dwconv2",
        )(x)
        x = layers.BatchNormalization(
            axis=BN_AXIS, momentum=self.bn_momentum, name=self.name + "bn"
        )(x)
        x = layers.Activation(self.activation, name=self.name + "activation")(x)

        # Squeeze and excite
        if 0 < self.se_ratio <= 1:
            filters_se = max(1, int(self.input_filters * self.se_ratio))
            se = layers.GlobalAveragePooling2D(name=self.name + "se_squeeze")(x)
            if BN_AXIS == 1:
                se_shape = (filters, 1, 1)
            else:
                se_shape = (1, 1, filters)
            se = layers.Reshape(se_shape, name=self.name + "se_reshape")(se)

            se = layers.Conv2D(
                filters_se,
                1,
                padding="same",
                activation=self.activation,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=self.name + "se_reduce",
            )(se)
            se = layers.Conv2D(
                filters,
                1,
                padding="same",
                activation="sigmoid",
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=self.name + "se_expand",
            )(se)

            x = layers.multiply([x, se], name=self.name + "se_excite")

            # Output phase
            x = layers.Conv2D(
                filters=self.output_filters,
                kernel_size=1,
                strides=1,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                padding="same",
                data_format="channels_last",
                use_bias=False,
                name=self.name + "project_conv",
            )(x)
            x = layers.BatchNormalization(
                axis=BN_AXIS, momentum=self.bn_momentum, name=self.name + "project_bn"
            )(x)

            if self.strides == 1 and self.input_filters == self.output_filters:
                if self.survival_probability:
                    x = layers.Dropout(
                        self.survival_probability,
                        noise_shape=(None, 1, 1, 1),
                        name=self.name + "drop",
                    )(x)
                x = layers.add([x, inputs], name=self.name + "add")
        return x
