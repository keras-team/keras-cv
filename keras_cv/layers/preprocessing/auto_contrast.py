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

import warnings

import tensorflow as tf

from keras_cv.utils import preprocessing


class AutoContrast(tf.keras.__internal__.layers.BaseImageAugmentationLayer):
    """Performs the AutoContrast operation on an image.

    Args:
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written [low, high].
            This is typically either `[0, 1]` or `[0, 255]` depending
            on how your preprocessing pipeline is setup.  Defaults to
            `[0, 255].`
    """

    def __init__(
        self,
        value_range=(0, 255),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.value_range = value_range
        # tf.vectorized_map can't handle iteration over a tf.range.
        self.auto_vectorize = False

    @staticmethod
    def scale_channel(image: tf.Tensor) -> tf.Tensor:
        """Scale the 2D image using the autocontrast rule."""
        low = tf.reduce_min(image)
        high = tf.reduce_max(image)

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(image):
            scale = 255.0 / (high - low)
            offset = -low * scale
            image = image * scale + offset
            image = tf.clip_by_value(image, 0.0, 255.0)
            return image

        # we can't scale uniform channels
        result = tf.cond(high > low, lambda: scale_values(image), lambda: image)
        return result

    def augment_image(self, image, transformation=None):
        image = preprocessing.transform_value_range(
            image, original_range=self.value_range, target_range=(0, 255)
        )
        channels = tf.shape(image)[-1]
        result = []
        for c in tf.range(channels):
            result.append(AutoContrast.scale_channel(image[..., c]))
        result = tf.stack(result, -1)
        result = preprocessing.transform_value_range(
            result, original_range=(0, 255), target_range=self.value_range
        )
        return result
