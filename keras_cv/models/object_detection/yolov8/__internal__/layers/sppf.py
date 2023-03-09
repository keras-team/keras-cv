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
class SPP(layers.Layer):
    """
    Performs Spatial Pyramid Pooling.
    """
    
    def __init__(self, filters, kernels=(5, 9, 13), **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernels = kernels
        self.num_filters = filters // 2
        self.conv1 = layers.Conv2D(
            filters=self.num_filters, kernel_size=1, strides=1
        )
        self.conv2 = layers.Conv2D(
            filters=self.filters, kernel_size=1, strides=1
        )
        self.modules = [
            layers.MaxPool2D(pool_size=x, strides=1, padding="SAME")
            for x in self.kernels
        ]

    def call(self, x):
        x = self.conv1(x)
        return self.conv2(
            tf.concat([x] + [module(x) for module in self.modules], axis=-1)
        )

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernels": self.kernels,
            "num_filters": self.num_filters,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class SPPF(layers.Layer):
    """
    Performs Spatial Pyramid Pooling.
    """
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, filters, kernel=5, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel = kernel
        self.num_filters = filters // 2
        self.conv1 = layers.Conv2D(
            filters=self.num_filters, kernel_size=1, strides=1
        )
        self.conv2 = layers.Conv2D(
            filters=self.filters, kernel_size=1, strides=1
        )
        self.module = layers.MaxPool2D(
            pool_size=self.kernel, strides=1, padding="SAME"
        )

    def call(self, x):
        x = self.conv1(x)
        x1 = self.module(x)
        x2 = self.module(x1)
        return self.conv2(tf.concat([x, x1, x2, self.module(x2)], axis=-1))

    def get_config(self):
        config = {
            "filters": self.filters,
            "kernel": self.kernel,
            "num_filters": self.num_filters,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
