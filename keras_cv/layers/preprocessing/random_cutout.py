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
from keras_cv.utils import fill_utils
from keras_cv.utils import preprocessing


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RandomCutout(BaseImageAugmentationLayer):
    """Randomly cut out rectangles from images and fill them.

    Args:
        height_factor: A tuple of two floats, a single float or a
            `keras_cv.FactorSampler`.  `height_factor` controls the size of the
            cutouts. `height_factor=0.0` means the rectangle will be of size 0% of the
            image height, `height_factor=0.1` means the rectangle will have a size of
            10% of the image height, and so forth.
            Values should be between `0.0` and `1.0`.  If a tuple is used, a
            `height_factor` is sampled between the two values for every image augmented.
            If a single float is used, a value between `0.0` and the passed float is
            sampled.  In order to ensure the value is always the same, please pass a
            tuple with two identical floats: `(0.5, 0.5)`.
        width_factor: A tuple of two floats, a single float or a
            `keras_cv.FactorSampler`.  `width_factor` controls the size of the
            cutouts. `width_factor=0.0` means the rectangle will be of size 0% of the
            image height, `width_factor=0.1` means the rectangle will have a size of 10%
            of the image width, and so forth.
            Values should be between `0.0` and `1.0`.  If a tuple is used, a
            `width_factor` is sampled between the two values for every image augmented.
            If a single float is used, a value between `0.0` and the passed float is
            sampled.  In order to ensure the value is always the same, please pass a
            tuple with two identical floats: `(0.5, 0.5)`.
        fill_mode: Pixels inside the patches are filled according to the given
            mode (one of `{"constant", "gaussian_noise"}`).
            - *constant*: Pixels are filled with the same constant value.
            - *gaussian_noise*: Pixels are filled with random gaussian noise.
        fill_value: a float represents the value to be filled inside the patches
            when `fill_mode="constant"`.
        seed: Integer. Used to create a random seed.

    Sample usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    random_cutout = keras_cv.layers.preprocessing.RandomCutout(0.5, 0.5)
    augmented_images = random_cutout(images)
    ```
    """

    def __init__(
        self,
        height_factor,
        width_factor,
        fill_mode="constant",
        fill_value=0.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)

        self.height_factor = preprocessing.parse_factor(
            height_factor, param_name="height_factor", seed=seed
        )
        self.width_factor = preprocessing.parse_factor(
            width_factor, param_name="width_factor", seed=seed
        )
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.seed = seed

        if fill_mode not in ["gaussian_noise", "constant"]:
            raise ValueError(
                '`fill_mode` should be "gaussian_noise" '
                f'or "constant".  Got `fill_mode`={fill_mode}'
            )

    def _parse_bounds(self, factor):
        if isinstance(factor, (tuple, list)):
            return factor[0], factor[1]
        else:
            return type(factor)(0), factor

    def get_random_transformation(self, image=None, **kwargs):
        center_x, center_y = self._compute_rectangle_position(image)
        rectangle_height, rectangle_width = self._compute_rectangle_size(image)
        return center_x, center_y, rectangle_height, rectangle_width

    def augment_image(self, image, transformation=None, **kwargs):
        """Apply random cutout."""
        inputs = tf.expand_dims(image, 0)
        center_x, center_y, rectangle_height, rectangle_width = transformation

        rectangle_fill = self._compute_rectangle_fill(inputs)
        inputs = fill_utils.fill_rectangle(
            inputs,
            center_x,
            center_y,
            rectangle_width,
            rectangle_height,
            rectangle_fill,
        )
        return inputs[0]

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def _compute_rectangle_position(self, inputs):
        input_shape = tf.shape(inputs)
        image_height, image_width = (
            input_shape[0],
            input_shape[1],
        )
        center_x = self._random_generator.random_uniform(
            [1], 0, image_width, dtype=tf.int32
        )
        center_y = self._random_generator.random_uniform(
            [1], 0, image_height, dtype=tf.int32
        )
        return center_x, center_y

    def _compute_rectangle_size(self, inputs):
        input_shape = tf.shape(inputs)
        image_height, image_width = (
            input_shape[0],
            input_shape[1],
        )
        height = self.height_factor()
        width = self.width_factor()

        height = height * tf.cast(image_height, tf.float32)
        width = width * tf.cast(image_width, tf.float32)

        height = tf.cast(tf.math.ceil(height), tf.int32)
        width = tf.cast(tf.math.ceil(width), tf.int32)

        height = tf.minimum(height, image_height)
        width = tf.minimum(width, image_width)

        return tf.expand_dims(height, axis=0), tf.expand_dims(width, axis=0)

    def _compute_rectangle_fill(self, inputs):
        input_shape = tf.shape(inputs)
        if self.fill_mode == "constant":
            fill_value = tf.fill(input_shape, self.fill_value)
        else:
            # gaussian noise
            fill_value = tf.random.normal(input_shape)

        return fill_value

    def get_config(self):
        config = {
            "height_factor": self.height_factor,
            "width_factor": self.width_factor,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
