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
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class SpatialPyramidPooling(layers.Layer):
    """
    Performs Spatial Pyramid Pooling.

    References:
        [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition]
        (https://arxiv.org/abs/1406.4729)

    Spatial Pyramid Pooling (SPP) is a layer used in convolutional neural networks (CNNs) 
    that enables them to accept images of different sizes instead of requiring a fixed 
    input image size. To achieve this, an SPP layer is added on top of the final 
    convolutional layer of the CNN. The SPP layer pools the features obtained from the 
    previous layer and generates fixed-length outputs, thereby removing the fixed-size 
    constraint of the network.

    Args:
        filters: int, the number of input filters
        kernel_sizes: int, kernel size or pool size for `MaxPool2D` layers, Defaults to (5, 9, 13).
    """

    def __init__(self, filters, kernel_sizes=(5, 9, 13), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.num_filters = filters // 2
        self.input_conv = layers.Conv2D(
            filters=self.num_filters, kernel_size=1, strides=1
        )
        self.output_conv = layers.Conv2D(
            filters=self.filters, kernel_size=1, strides=1
        )
        self.modules = [
            layers.MaxPool2D(pool_size=x, strides=1, padding="SAME")
            for x in self.kernel_sizes
        ]

    def call(self, x):
        x = self.input_conv(x)
        return self.output_conv(
            tf.concat([x] + [module(x) for module in self.modules], axis=-1)
        )

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernels": self.kernel_sizes,
            "num_filters": self.num_filters,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class FastSpatialPyramidPooling(layers.Layer):
    """
    Performs Spatial Pyramid Pooling Faster.

    SPPF(Spatial Pyramid Pooling Fast) is an optimized version of SPP, which has less
    FLOPS(loating point operations per second).
    
    Args:
        filters: int, the number of input filters
        kernel_sizes: int, kernel size or pool size for `MaxPool2D` layers, Defaults to 5.
    """

    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, filters, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_filters = filters // 2
        self.input_conv = layers.Conv2D(
            filters=self.num_filters, kernel_size=1, strides=1
        )
        self.output_conv = layers.Conv2D(
            filters=self.filters, kernel_size=1, strides=1
        )
        self.max_pool = layers.MaxPool2D(
            pool_size=self.kernel_size, strides=1, padding="SAME"
        )

    def call(self, x):
        x = self.input_conv(x)
        x1 = self.max_pool(x)
        x2 = self.max_pool(x1)
        return self.output_conv(
            tf.concat([x, x1, x2, self.max_pool(x2)], axis=-1)
        )

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel": self.kernel_size,
            "num_filters": self.num_filters,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
