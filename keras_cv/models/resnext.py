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
from tensorflow.keras import backend
from tensorflow.keras import layers
import math

from keras_cv.models import utils

class GroupConv2D(tf.keras.layers.Layer):
    def __init__(self, input_channels, output_channels, groups, kernel_size, bottleneck_width, strides=1, padding='valid', **kwargs):
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
            self.conv_list.append(tf.keras.layers.Conv2D(filters=self.group_width, kernel_size=kernel_size, strides=strides,
                                                         padding=padding,
                                                         **kwargs))

    def call(self, inputs, **kwargs):
        feature_map_list = []
        for i in range(self.groups):
            x_i = self.conv_list[i](inputs[:, :, :, i*self.group_in_num: (i + 1) * self.group_in_num])
            feature_map_list.append(x_i)
        out = tf.concat(feature_map_list, axis=-1)
        return out                                                     


def ConvBlock(inputs, filters, kernel_size, strides, padding):
    x = tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding=padding)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    return x

def ResNeXt_BottleNeck(inputs, filters, strides, groups, bottleneck_width):
    x = ConvBlock(inputs, filters, (1,1), 1, "same")
    x = GroupConv2D(input_channels=filters,output_channels=filters, kernel_size=(3, 3),
                    strides=strides,padding="same",groups=groups, bottleneck_width=bottleneck_width)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = ConvBlock(x, 2 * filters, (1,1), 1, "same")
    shortcut = ConvBlock(inputs, filters = 2 * filters, kernel_size=(1,1), strides=strides, padding="same")
    outputs = tf.keras.layers.add([x, shortcut])
    return outputs

def ResNeXt_Block(inputs, filters, strides, groups, num_blocks, bottleneck_width):
    x = ResNeXt_BottleNeck(inputs=inputs, filters=filters, strides=strides, groups=groups, bottleneck_width=bottleneck_width)
    for _ in range(1, num_blocks):
        x = ResNeXt_BottleNeck(x, filters=filters, strides=1, groups=groups, bottleneck_width=bottleneck_width)

    return x

def ResNeXt(num_blocks, cardinality, bottleneck_width, input_shape=(224, 224, 3), input_tensor=None, classes=1):
    inputs = utils.parse_model_inputs(input_shape, input_tensor)
    x = inputs
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")(x)

    x = ResNeXt_Block(x, filters=128, strides=1, groups=cardinality, num_blocks=num_blocks[0], bottleneck_width=bottleneck_width)
    x = ResNeXt_Block(x, filters=256, strides=2, groups=cardinality, num_blocks=num_blocks[1], bottleneck_width=bottleneck_width)
    x = ResNeXt_Block(x, filters=512, strides=2, groups=cardinality, num_blocks=num_blocks[2], bottleneck_width=bottleneck_width)
    x = ResNeXt_Block(x, filters=1024, strides=2, groups=cardinality, num_blocks=num_blocks[3], bottleneck_width=bottleneck_width)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=classes, activation='softmax')(x)
    model = tf.keras.Model(inputs = inputs,outputs=x)
    return model

def resnext50_32x4d():
    return ResNeXt(num_blocks=[3, 4, 6, 3],
                   cardinality=32, bottleneck_width=4)


def resnext101_32x4d():
    return ResNeXt(num_blocks=[3, 4, 23, 3],
                   cardinality=32, bottleneck_width=4)


def resnext101_64x4d():
    return ResNeXt(num_blocks=[3, 4, 23, 3],
                   cardinality=64, bottleneck_width=4)


model = resnext101_64x4d()
print(model.summary())