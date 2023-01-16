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


class ImplicitAddition(tf.keras.layers.Layer):
    def __init__(self, mean=0.0, std=0.02, **kwargs):
        super(ImplicitAddition, self).__init__(**kwargs)
        self.mean = mean
        self.std = std

    def build(self, input_shape):
        self.implicit = tf.Variable(
            initial_value=tf.random_normal_initializer(mean=self.mean, stddev=self.std)(
                shape=(1, 1, 1, input_shape[-1])
            ),
            trainable=True,
            name=self.name,
        )

    def call(self, x):
        return tf.cast(x, self.implicit.dtype) + self.implicit

    def get_config(self):
        config = super(ImplicitAddition, self).get_config()
        config.update({"mean": self.mean, "std": self.std})
        return config
