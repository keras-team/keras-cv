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


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RandomSnow(tf.keras.__internal__.layers.BaseImageAugmentationLayer):
    """Randomly adjusts the hue on given images.

    This layer will randomly add snow effect on the input RGB
    images. At inference time, the output will be identical to the input.

    Snow is added on the image by converting image to HSV and rotating
    channel (V) in order to get snow effect on image.

    Args:
        factor: A tuple of two floats, a single float or `keras_cv.FactorSampler`.
            `factor` controls the extent to which the snow effect is impacted.
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
        self.brightness_coefficient = tf.constant(2.5)

    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        return self.factor()

    def augment_image(self, image, transformation=None):
        image = preprocessing.transform_value_range(image, self.value_range, (0, 1))

        image_HSV = tf.image.rgb_to_hsv(image)
        image_HSV = tf.cast(image_HSV, tf.float64)

        # scale transformation to find snow points.
        transformation *= 255 / 2
        transformation += 255 / 3

        # Saturation or channel (S).
        image_S = image_HSV[:, :, 1]

        # Value or Brightness channel.
        image_V = image_HSV[:, :, 2]

        # Calculate lightness in HSL color space and scale.
        image_L = (
            tf.math.reduce_min(image, axis=-1) + tf.math.reduce_max(image, axis=-1)
        ) / tf.constant(2, tf.float32)
        image_L = tf.where(
            image_L < transformation, image_L * self.brightness_coefficient, image_L
        )
        image_L = tf.clip_by_value(image_L, clip_value_min=0, clip_value_max=255)

        # Get corresponding channel (V) in HSV space.
        image_V = image_L / (1 - (image_S / tf.constant(2, tf.float32)))
        image_V = tf.clip_by_value(image_V, clip_value_min=0, clip_value_max=255)

        # Get corresponding channel (S) in HSV space.
        image_S = tf.where(
            image_V == 0, 0.0, tf.constant(2.0) * (tf.constant(1.0) - image_L / image_V)
        )
        image_S = tf.clip_by_value(image_S, clip_value_min=0, clip_value_max=1.0)

        image_HSV = tf.stack([image_HSV[:, :, 0], image_S, image_V], axis=-1)
        image_RGB = tf.image.hsv_to_rgb(image_HSV)
        image = preprocessing.transform_value_range(image_RGB, (0, 1), self.value_range)
        return image

    def augment_label(self, label, transformation=None):
        return label

    def get_config(self):
        config = {
            "factor": self.factor,
            "value_range": self.value_range,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
