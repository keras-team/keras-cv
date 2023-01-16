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


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class FusedConvolution(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size=1,
        strides=1,
        padding="same",
        groups=1,
        act=True,
        deploy=False,
        **kwargs,
    ):
        super(FusedConvolution, self).__init__(**kwargs)
        self.deploy = deploy
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.groups = groups
        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            groups=groups,
            use_bias=False,
        )
        self.bn = None
        if not deploy:
            self.bn = tf.keras.layers.BatchNormalization()
        self.act = (
            tf.keras.activations.swish
            if act is True
            else (act if isinstance(act, tf.keras.activations) else tf.identity)
        )

    def call(self, x):
        if self.deploy:
            return self.act(self.conv(x))
        else:
            return self.act(self.bn(self.conv(x)))

    def get_config(self):
        config = super(FusedConvolution, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "padding": self.padding,
                "strides": self.strides,
                "groups": self.groups,
            }
        )
        return config
