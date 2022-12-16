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

import math

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class GroupConv2D(layers.Layer):
    def __init__(
        self,
        input_channels,
        output_channels,
        groups,
        kernel_size,
        bottleneck_width,
        strides=1,
        padding="valid",
        **kwargs,
    ):
        super(GroupConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.groups = groups
        self.bottleneck_width = bottleneck_width

        self.mid_channels = output_channels // 4
        D = int(math.floor(self.mid_channels * (self.bottleneck_width / 64.0)))
        self.group_width = self.groups * D
        self.group_in_num = input_channels // self.groups
        self.group_out_num = output_channels // self.groups
        self.conv_list = []
        for i in range(self.groups):
            self.conv_list.append(
                tf.keras.layers.Conv2D(
                    filters=self.group_width,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    **kwargs,
                )
            )

    def call(self, inputs, **kwargs):
        feature_map_list = []
        for i in range(self.groups):
            x_i = self.conv_list[i](
                inputs[:, :, :, i * self.group_in_num : (i + 1) * self.group_in_num]
            )
            feature_map_list.append(x_i)
        out = tf.concat(feature_map_list, axis=-1)
        return out