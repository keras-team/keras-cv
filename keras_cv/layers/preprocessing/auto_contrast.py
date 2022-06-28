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
from keras_cv.utils import preprocessing


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class AutoContrast(BaseImageAugmentationLayer):
    """Performs the AutoContrast operation on an image.

    Auto contrast stretches the values of an image across the entire available
    `value_range`.  This makes differences between pixels more obvious.  An example of
    this is if an image only has values `[0, 1]` out of the range `[0, 255]`, auto
    contrast will change the `1` values to be `255`.

    Args:
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written [low, high].
            This is typically either `[0, 1]` or `[0, 255]` depending
            on how your preprocessing pipeline is setup.
    """

    def __init__(
        self,
        value_range,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.value_range = value_range

    def augment_image(self, image, transformation=None, **kwargs):
        original_image = image
        image = preprocessing.transform_value_range(
            image, original_range=self.value_range, target_range=(0, 255)
        )

        low = tf.reduce_min(tf.reduce_min(image, axis=0), axis=0)
        high = tf.reduce_max(tf.reduce_max(image, axis=0), axis=0)
        scale = 255.0 / (high - low)
        offset = -low * scale

        image = image * scale[None, None] + offset[None, None]
        result = tf.clip_by_value(image, 0.0, 255.0)
        result = preprocessing.transform_value_range(
            result, original_range=(0, 255), target_range=self.value_range
        )
        # don't process NaN channels
        result = tf.where(tf.math.is_nan(result), original_image, result)
        return result

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def get_config(self):
        config = super().get_config()
        config.update({"value_range": self.value_range})
        return config
