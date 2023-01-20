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
class FusedMBConvBlock(layers.Layer):
    """
    Implementation of the FusedMBConv block (Fused Mobile Inverted Residual Bottleneck) from:
        (EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML)[https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html]
        (EfficientNetV2: Smaller Models and Faster Training)[https://arxiv.org/abs/2104.00298v3].

    FusedMBConv blocks are based on MBConv blocks, and replace the depthwise and 1x1 output convolution
    blocks with a single 3x3 convolution block, fusing them together - hence the name "FusedMBConv".
    Alongside MBConv blocks, they can be used in mobile-oriented and efficient architectures,
    and are present in architectures EfficientNet.

    FusedMBConv blocks follow a narrow-wide-narrow structure - expanding a 1x1 convolution, performing
    Squeeze-Excitation and then applying a 3x3 convolution, which is a more efficient operation than
    conventional wide-narrow-wide structures.

    As they're frequently used for models to be deployed to edge devices, they're
    implemented as a layer for ease of use and re-use.

    Args:
        input_filters: int, the number of input filters
        output_filters: int, the number of output filters
        expand_ratio: default 1, the ratio by which input_filters are multiplied to expand
            the structure in the middle expansion phase
        kernel_size: default 3, the kernel_size to apply to the expansion phase convolutions
        strides: default 1, the strides to apply to the expansion phase convolutions
        se_ratio: default 0.0, The filters used in the Squeeze-Excitation phase, and are chosen as
            the maximum between 1 and input_filters*se_ratio
        bn_momentum: default 0.9, the BatchNormalization momentum
        activation: default "swish", the activation function used between convolution operations
        survival_probability: float, default 0.8, the optional dropout rate to apply before the output
            convolution

    Returns:
        A `tf.Tensor` representing a feature map, passed through the FusedMBConv block


    Example usage:

    ```
    inputs = tf.random.normal(shape=(1, 64, 64, 32), dtype=tf.float32)
    layer = keras_cv.layers.FusedMBConvBlock(input_filters=32, output_filters=32)
    output = layer(inputs)
    output.shape # TensorShape([1, 224, 224, 48])
    ```
    """

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
        **kwargs
    ):

        super().__init__(**kwargs)
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.strides = strides
        self.se_ratio = se_ratio
        self.bn_momentum = bn_momentum
        self.activation = activation
        self.survival_probability = survival_probability
        self.filters = self.input_filters * self.expand_ratio
        self.filters_se = max(1, int(input_filters * se_ratio))

        self.conv1 = layers.Conv2D(
            filters=self.filters,
            kernel_size=kernel_size,
            strides=strides,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            data_format="channels_last",
            use_bias=False,
            name=self.name + "expand_conv",
        )
        self.bn1 = layers.BatchNormalization(
            axis=BN_AXIS,
            momentum=self.bn_momentum,
            name=self.name + "expand_bn",
        )
        self.act = layers.Activation(
            self.activation, name=self.name + "expand_activation"
        )

        self.bn2 = layers.BatchNormalization(
            axis=BN_AXIS, momentum=self.bn_momentum, name=self.name + "bn"
        )

        self.se_conv1 = layers.Conv2D(
            self.filters_se,
            1,
            padding="same",
            activation=self.activation,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=self.name + "se_reduce",
        )

        self.se_conv2 = layers.Conv2D(
            self.filters,
            1,
            padding="same",
            activation="sigmoid",
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=self.name + "se_expand",
        )

        self.output_conv = layers.Conv2D(
            filters=self.output_filters,
            kernel_size=1 if expand_ratio != 1 else kernel_size,
            strides=1,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            data_format="channels_last",
            use_bias=False,
            name=self.name + "project_conv",
        )

        self.bn3 = layers.BatchNormalization(
            axis=BN_AXIS, momentum=self.bn_momentum, name=self.name + "project_bn"
        )

    def build(self, input_shape):
        if self.name is None:
            self.name = backend.get_uid("block0")

    def call(self, inputs):
        # Expansion phase
        if self.expand_ratio != 1:
            x = self.conv1(inputs)
            x = self.bn1(x)
            x = self.act(x)
        else:
            x = inputs

        # Squeeze and excite
        if 0 < self.se_ratio <= 1:
            se = layers.GlobalAveragePooling2D(name=self.name + "se_squeeze")(x)
            if BN_AXIS == 1:
                se_shape = (self.filters, 1, 1)
            else:
                se_shape = (1, 1, self.filters)

            se = layers.Reshape(se_shape, name=self.name + "se_reshape")(se)

            se = self.se_conv1(se)
            se = self.se_conv2(se)

            x = layers.multiply([x, se], name=self.name + "se_excite")

        # Output phase:
        x = self.output_conv(x)
        x = self.bn3(x)
        if self.expand_ratio == 1:
            x = self.act(x)

        # Residual:
        if self.strides == 1 and self.input_filters == self.output_filters:
            if self.survival_probability:
                x = layers.Dropout(
                    self.survival_probability,
                    noise_shape=(None, 1, 1, 1),
                    name=self.name + "drop",
                )(x)
            x = layers.add([x, inputs], name=self.name + "add")
        return x

    def get_config(self):
        config = {
            "input_filters": self.input_filters,
            "output_filters": self.output_filters,
            "expand_ratio": self.expand_ratio,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "se_ratio": self.se_ratio,
            "bn_momentum": self.bn_momentum,
            "activation": self.activation,
            "survival_probability": self.survival_probability,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
