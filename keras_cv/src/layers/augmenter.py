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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras


@keras_cv_export("keras_cv.layers.Augmenter")
class Augmenter(keras.layers.Layer):
    """Light-weight class to apply augmentations to data.

    Args:
        layers: A list of `keras.layers.Layers` to apply to the example

    Example:
    from keras_cv.src import layers
    images = np.ones((16, 256, 256, 3))
    augmenter = layers.Augmenter(
        [
            layers.RandomFlip(),
            layers.RandAugment(value_range=(0, 255)),
            layers.CutMix(),
        ]
    )
    augmented_images = augmenter(images)
    """

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
