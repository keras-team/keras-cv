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

from keras_cv.utils import preprocessing


class RandomColorDegeneration(tf.keras.__internal__.layers.BaseImageAugmentationLayer):
    """Randomly performs the color degeneration operation on given images.

    The sharpness operation first converts an image to gray scale, then back to color.
    It then takes a weighted average between original image and the degenerated image.
    This makes colors appear more dull.

    Args:
        factor: Either a tuple of two floats or a single float. `factor` controls the
            extent to which the image sharpness is impacted.  `factor=0.0` makes this
            layer perform a no-op operation, while a value of 1.0 uses the degenerated
            result entirely.  Values between 0 and 1 result in linear interpolation
            between the original image and the sharpened image.

            Values should be between `0.0` and `1.0`.  If a tuple is used, a `factor` is
            sampled between the two values for every image augmented.  If a single float
            is used, a value between `0.0` and the passed float is sampled.  In order to
            ensure the value is always the same, please pass a tuple with two identical
            floats: `(0.5, 0.5)`.
    """

    def __init__(
        self,
        factor,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.factor = preprocessing.parse_factor_value_range(factor)

    def get_random_transformation(self):
        if self.factor[0] == self.factor[1]:
            return self.factor[0]
        return self._random_generator.random_uniform(
            (), self.factor[0], self.factor[1], dtype=tf.float32
        )

    def augment_image(self, image, transformation=None):
        degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
        result = preprocessing.blend(image, degenerate, transformation)
        return result
