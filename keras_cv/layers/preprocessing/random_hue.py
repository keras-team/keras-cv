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
class RandomHue(BaseImageAugmentationLayer):
    """Randomly adjusts the hue on given images.

    This layer will randomly increase/reduce the hue for the input RGB
    images. At inference time, the output will be identical to the input.
    Call the layer with `training=True` to adjust the brightness of the input.

    The image hue is adjusted by converting the image(s) to HSV and rotating the
    hue channel (H) by delta. The image is then converted back to RGB.

    Args:
        factor: A tuple of two floats, a single float or `keras_cv.FactorSampler`.
            `factor` controls the extent to which the image hue is impacted.
            `factor=0.0` makes this layer perform a no-op operation, while a value of
            1.0 performs the most aggressive contrast adjustment available.  If a tuple
            is used, a `factor` is sampled between the two values for every image
            augmented.  If a single float is used, a value between `0.0` and the passed
            float is sampled.  In order to ensure the value is always the same, please
            pass a tuple with two identical floats: `(0.5, 0.5)`.
        value_range:  the range of values the incoming images will have.
            Represented as a two number tuple written [low, high].
            This is typically either `[0, 1]` or `[0, 255]` depending
            on how your preprocessing pipeline is setup.
        seed: Integer. Used to create a random seed.

    """

    def __init__(self, factor, value_range, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.factor = preprocessing.parse_factor(
            factor,
        )
        self.value_range = value_range
        self.seed = seed

    def get_random_transformation(self, **kwargs):
        invert = preprocessing.random_inversion(self._random_generator)
        # We must scale self.factor() to the range [-0.5, 0.5].  This is because the
        # tf.image operation performs rotation on the hue saturation value orientation.
        # This can be thought of as an angle in the range [-180, 180]
        return invert * self.factor() * 0.5

    def augment_image(self, image, transformation=None, **kwargs):
        image = preprocessing.transform_value_range(image, self.value_range, (0, 1))
        # tf.image.adjust_hue expects floats to be in range [0, 1]
        image = tf.image.adjust_hue(image, delta=transformation)
        # RandomHue is one of the rare KPLs that needs to clip
        image = tf.clip_by_value(image, 0, 1)
        image = preprocessing.transform_value_range(image, (0, 1), self.value_range)
        return image

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def get_config(self):
        config = {
            "factor": self.factor,
            "value_range": self.value_range,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
