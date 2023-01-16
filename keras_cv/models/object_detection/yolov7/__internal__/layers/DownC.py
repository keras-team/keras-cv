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
import FusedConvolution


class DownC(tf.keras.layers.Layer):
    def __init__(self, filters, n=1, kernel_size=2, **kwargs):
        super(DownC, self).__init__(**kwargs)
        self.filters = filters
        self.n = n
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.cv1 = FusedConvolution(input_shape[-1], 1, 1)
        self.cv2 = FusedConvolution(self.filters // 2, 3, self.kernel_size)
        self.cv3 = FusedConvolution(self.filters // 2, 1, 1)
        self.mp = tf.keras.layers.MaxPooling2D(
            pool_size=self.kernel_size, strides=self.kernel_size
        )

    def call(self, x):
        return tf.concat([self.cv2(self.cv1(x)), self.cv3(self.mp(x))], -1)

    def get_config(self):
        config = super(DownC, self).get_config()
        config.update(
            {"filters": self.filters, "n": self.n, "kernel_size": self.kernel_size}
        )
        return config
