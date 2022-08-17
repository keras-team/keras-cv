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
class Augmenter(tf.keras.layers.Layer):
    def __init__(self, augmentation_layers, **kwargs):
        super().__init__(**kwargs)
        self.augmentation_layers = augmentation_layers

    def call(self, inputs):
        for layer in self.augmentation_layers:
            inputs = layer(inputs)
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"augmentation_layers": self.augmentation_layers})
        return config
