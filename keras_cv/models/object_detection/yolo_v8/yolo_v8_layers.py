# Copyright 2023 The KerasCV Authors
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
from keras import layers

BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_MOMENTUM = 0.97


def conv_bn(
    inputs,
    output_channel,
    kernel_size=1,
    strides=1,
    activation="swish",
    name=None,
):
    if kernel_size > 1:
        inputs = layers.ZeroPadding2D(
            padding=kernel_size // 2, name=f"{name}_pad"
        )(inputs)

    nn = layers.Conv2D(
        filters=output_channel,
        kernel_size=kernel_size,
        strides=strides,
        padding="valid",
        use_bias=False,
        name=f"{name}_conv",
    )(inputs)
    nn = layers.BatchNormalization(
        momentum=BATCH_NORM_MOMENTUM,
        epsilon=BATCH_NORM_EPSILON,
        name=f"{name}_bn",
    )(nn)
    nn = layers.Activation(activation, name=name)(nn)
    return nn


def csp_with_2_conv(
    inputs,
    channels=-1,
    depth=2,
    shortcut=True,
    expansion=0.5,
    activation="swish",
    name=None,
):
    channel_axis = -1
    channels = channels if channels > 0 else inputs.shape[channel_axis]
    hidden_channels = int(channels * expansion)

    pre = conv_bn(
        inputs,
        hidden_channels * 2,
        kernel_size=1,
        activation=activation,
        name=f"{name}_pre",
    )
    short, deep = tf.split(pre, 2, axis=channel_axis)

    out = [short, deep]
    for id in range(depth):
        deep = conv_bn(
            deep,
            hidden_channels,
            kernel_size=3,
            activation=activation,
            name=f"{name}_pre_{id}_1",
        )
        deep = conv_bn(
            deep,
            hidden_channels,
            kernel_size=3,
            activation=activation,
            name=f"{name}_pre_{id}_2",
        )
        deep = (out[-1] + deep) if shortcut else deep
        out.append(deep)
    out = tf.concat(out, axis=channel_axis)
    out = conv_bn(
        out,
        channels,
        kernel_size=1,
        activation=activation,
        name=f"{name}_output",
    )
    return out
