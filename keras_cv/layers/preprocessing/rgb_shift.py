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


class RGBShift(tf.keras.__internal__.layers.BaseImageAugmentationLayer):
    """Randomly shift values for each channel of the input image(s).

    The input images should have values in the `[0-255]` or `[0-1]` range.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `channels_last` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `channels_last` format.

    Args:
        factor: A scalar value, or tuple/list of two floating values in
            the range `[0.0, 1.0]`. If `factor` is a single value, it will
            interpret as equivalent to the tuple `(0.0, factor)`. The `factor` 
            will sampled between its range for every image to augment. 
        value_range: The range of values the incoming images will have.
            Represented as a two number tuple written [low, high].
            This is typically either `[0, 1]` or `[0, 255]` depending
            on how your preprocessing pipeline is setup.
        seed: Integer. Used to create a random seed.

    Usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    rgb_shift = keras_cv.layers.RGBShift(factor=(0.3, 0.8), value_range=(0, 255))
    augmented_images = rgb_shift(images)
    ```
    """

    def __init__(self, factor, value_range, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.value_range = value_range
        self.factor = preprocessing.parse_factor(factor, seed=self.seed)

    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        invert = preprocessing.random_inversion(self._random_generator)
        r_shift = invert * self.factor() * 0.5

        invert = preprocessing.random_inversion(self._random_generator)
        g_shift = invert * self.factor() * 0.5

        invert = preprocessing.random_inversion(self._random_generator)
        b_shift = invert * self.factor() * 0.5

        return [r_shift, g_shift, b_shift]

    def augment_image(self, image, transformation=None):
        image = preprocessing.transform_value_range(image, self.value_range, (0, 1))
        unstack_rgb = tf.unstack(tf.cast(image, tf.float32), axis=-1)
        shifted_rgb = tf.stack(
            [
                tf.add(unstack_rgb[0], transformation[0]),
                tf.add(unstack_rgb[1], transformation[1]),
                tf.add(unstack_rgb[2], transformation[2]),
            ],
            axis=-1,
        )
        shifted_rgb = tf.clip_by_value(shifted_rgb, 0.0, 1.0)
        image = preprocessing.transform_value_range(
            shifted_rgb, (0, 1), self.value_range
        )
        return image

    def get_config(self):
        config = super().get_config()
        config.update(
            {"factor": self.factor, "value_range": self.value_range, "seed": self.seed}
        )
        return config
