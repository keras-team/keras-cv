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

from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class Augmenter(BaseImageAugmentationLayer):
    """Augmenter performs a series of preprocessing operations on input data.
    Args:
        layers: A list of Keras layers to be applied in sequence to input data.
    """

    def __init__(self, layers, **kwargs):
        super().__init__(**kwargs)
        self.layers = layers

    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"layers": self.layers})
        return config
