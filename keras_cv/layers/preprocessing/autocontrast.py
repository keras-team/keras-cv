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


class AutoContrast(tf.keras.__internal__.layers.BaseImageAugmentationLayer):
    """Performs the AutoContrast operation on an image.

    Args:
        value_range
    """

    def __init__(
        self,
        value_range=(0, 255)
        **kwargs,
    ):
        super().__init__(**kwargs)

    @staticmethod
    def scale_channel(image: tf.Tensor) -> tf.Tensor:
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = tf.cast(tf.reduce_min(image), tf.float32)
        hi = tf.cast(tf.reduce_max(image), tf.float32)

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = tf.cast(im, tf.float32) * scale + offset
            im = tf.clip_by_value(im, 0.0, 255.0)
            return tf.cast(im, tf.uint8)

        result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
        return result

    def augment_image(self, image, transformation=None):
        image = #
        channels = tf.shape(image)[-1]
        result = []
        for c in channels:
            result.append(AutoContrast.scale_channel(image[..., c]))
        result = tf.stack(result, -1)
