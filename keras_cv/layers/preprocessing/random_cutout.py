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

from keras_cv.api_export import keras_cv_export
from keras_cv.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    VectorizedBaseImageAugmentationLayer,
)
from keras_cv.utils import fill_utils
from keras_cv.utils import preprocessing


@keras_cv_export("keras_cv.layers.RandomCutout")
class RandomCutout(VectorizedBaseImageAugmentationLayer):
    """Randomly cut out rectangles from images and fill them.

    Args:
        height_factor: A tuple of two floats, a single float or a
            `keras_cv.FactorSampler`. `height_factor` controls the size of the
            cutouts. `height_factor=0.0` means the rectangle will be of size 0%
            of the image height, `height_factor=0.1` means the rectangle will
            have a size of 10% of the image height, and so forth. Values should
            be between `0.0` and `1.0`. If a tuple is used, a `height_factor`
            is sampled between the two values for every image augmented. If a
            single float is used, a value between `0.0` and the passed float is
            sampled. In order to ensure the value is always the same, please
            pass a tuple with two identical floats: `(0.5, 0.5)`.
        width_factor: A tuple of two floats, a single float or a
            `keras_cv.FactorSampler`. `width_factor` controls the size of the
            cutouts. `width_factor=0.0` means the rectangle will be of size 0%
            of the image height, `width_factor=0.1` means the rectangle will
            have a size of 10% of the image width, and so forth.
            Values should be between `0.0` and `1.0`. If a tuple is used, a
            `width_factor` is sampled between the two values for every image
            augmented. If a single float is used, a value between `0.0` and the
            passed float is sampled. In order to ensure the value is always the
            same, please pass a tuple with two identical floats: `(0.5, 0.5)`.
        fill_mode: Pixels inside the patches are filled according to the given
            mode (one of `{"constant", "gaussian_noise"}`).
            - *constant*: Pixels are filled with the same constant value.
            - *gaussian_noise*: Pixels are filled with random gaussian noise.
        fill_value: a float represents the value to be filled inside the patches
            when `fill_mode="constant"`.
        seed: Integer. Used to create a random seed.

    Sample usage:
    ```python
    (images, labels), _ = keras.datasets.cifar10.load_data()
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
                f'or "constant". Got `fill_mode`={fill_mode}'
            )

    def get_random_transformation_batch(self, batch_size, images, **kwargs):
        centers_x, centers_y = self._compute_rectangle_position(images)
        rectangles_height, rectangles_width = self._compute_rectangle_size(images)
        return centers_x, centers_y, rectangles_height, rectangles_width

    def augment_images(self, images, transformations=None, **kwargs):
        """Apply random cutout."""
        centers_x, centers_y, rectangles_height, rectangles_width = transformations

        rectangles_fill = self._compute_rectangle_fill(images)
        images = fill_utils.fill_rectangle(
            images,
            centers_x,
            centers_y,
            rectangles_width,
            rectangles_height,
            rectangles_fill,
        )
        return images

    def augment_bounding_boxes(self, bounding_boxes, **kwargs):
        return bounding_boxes

    def augment_labels(self, labels, transformations=None, **kwargs):
        return labels

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        return segmentation_masks

    def augment_keypoints(self, keypoints, transformations, **kwargs):
        return keypoints

    def augment_targets(self, targets, transformations, **kwargs):
        return targets

    def augment_ragged_image(self, image, transformation, **kwargs):
        return self.augment_images(
            image, transformations=transformation, **kwargs
        )

    def _compute_rectangle_position(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, image_height, image_width = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )
        center_x = self._random_generator.uniform(
            [batch_size], 0, image_width, dtype=tf.int32
        )
        center_y = self._random_generator.uniform(
            [batch_size], 0, image_height, dtype=tf.int32
        )
        return center_x, center_y

    def _compute_rectangle_size(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, image_height, image_width = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )
        height = self.height_factor(shape=(batch_size,))
        width = self.width_factor(shape=(batch_size,))

        height = height * tf.cast(image_height, tf.float32)
        width = width * tf.cast(image_width, tf.float32)

        height = tf.cast(tf.math.ceil(height), tf.int32)
        width = tf.cast(tf.math.ceil(width), tf.int32)

        height = tf.minimum(height, image_height)
        width = tf.minimum(width, image_width)

        return height, width

    def _compute_rectangle_fill(self, inputs):
        input_shape = tf.shape(inputs)
        if self.fill_mode == "constant":
            fill_value = tf.fill(input_shape, self.fill_value)
            fill_value = tf.cast(fill_value, dtype=self.compute_dtype)
        else:
            # gaussian noise
            fill_value = tf.random.normal(input_shape, dtype=self.compute_dtype)
            # rescale the random noise to the original image range
            image_max = tf.reduce_max(inputs)
            image_min = tf.reduce_min(inputs)
            fill_max = tf.reduce_max(fill_value)
            fill_min = tf.reduce_min(fill_value)
            fill_value = (image_max - image_min) * (fill_value - fill_min) / (fill_max - fill_min) + image_min
        return fill_value

    def get_config(self):
        config = super().get_config()
        config.update({
            "height_factor": self.height_factor,
            "width_factor": self.width_factor,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "seed": self.seed,
        })
        return config
