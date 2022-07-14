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
class RandomColorDegeneration(BaseImageAugmentationLayer):
    """Randomly performs the color degeneration operation on given images.

    The sharpness operation first converts an image to gray scale, then back to color.
    It then takes a weighted average between original image and the degenerated image.
    This makes colors appear more dull.

    Args:
        factor: A tuple of two floats, a single float or a
            `keras_cv.FactorSampler`. `factor` controls the extent to which the
            image sharpness is impacted. `factor=0.0` makes this layer perform a no-op
            operation, while a value of 1.0 uses the degenerated result entirely.
            Values between 0 and 1 result in linear interpolation between the original
            image and the sharpened image.
            Values should be between `0.0` and `1.0`.  If a tuple is used, a `factor` is
            sampled between the two values for every image augmented.  If a single float
            is used, a value between `0.0` and the passed float is sampled.  In order to
            ensure the value is always the same, please pass a tuple with two identical
            floats: `(0.5, 0.5)`.
        seed: Integer. Used to create a random seed.
    """

    def __init__(
        self,
        factor,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.factor = preprocessing.parse_factor(
            factor,
        )
        self.seed = seed

    def get_random_transformation(self, **kwargs):
        return self.factor()

    def augment_image(self, image, transformation=None, **kwargs):
        degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
        result = preprocessing.blend(image, degenerate, transformation)
        return result

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def get_config(self):
        config = super().get_config()
        config.update({"factor": self.factor, "seed": self.seed})
        return config
