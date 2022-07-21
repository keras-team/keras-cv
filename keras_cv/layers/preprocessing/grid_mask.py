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
from tensorflow.keras import layers

from keras_cv import core
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.utils import fill_utils
from keras_cv.utils import preprocessing


def _center_crop(mask, width, height):
    masks_shape = tf.shape(mask)
    h_diff = masks_shape[0] - height
    w_diff = masks_shape[1] - width

    h_start = tf.cast(h_diff / 2, tf.int32)
    w_start = tf.cast(w_diff / 2, tf.int32)
    return tf.image.crop_to_bounding_box(mask, h_start, w_start, height, width)


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class GridMask(BaseImageAugmentationLayer):
    """GridMask class for grid-mask augmentation.


    Input shape:
        Int or float tensor with values in the range [0, 255].
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format

    Args:
        ratio_factor: A float, tuple of two floats, or `keras_cv.FactorSampler`.
            Ratio determines the ratio from spacings to grid masks.
            Lower values make the grid
            size smaller, and higher values make the grid mask large.
            Floats should be in the range [0, 1].  0.5 indicates that grid and
            spacing will be of equal size. To always use the same value, pass a
            `keras_cv.ConstantFactorSampler()`.

            Defaults to `(0, 0.5)`.
        rotation_factor:
            The rotation_factor will be used to randomly rotate the grid_mask during
            training. Default to 0.1, which results in an output rotating by a
            random amount in the range [-10% * 2pi, 10% * 2pi].

            A float represented as fraction of 2 Pi, or a tuple of size 2
            representing lower and upper bound for rotating clockwise and
            counter-clockwise. A positive values means rotating counter clock-wise,
            while a negative value means clock-wise. When represented as a single
            float, this value is used for both the upper and lower bound. For
            instance, factor=(-0.2, 0.3) results in an output rotation by a random
            amount in the range [-20% * 2pi, 30% * 2pi]. factor=0.2 results in an
            output rotating by a random amount in the range [-20% * 2pi, 20% * 2pi].

        fill_mode: Pixels inside the gridblock are filled according to the given
            mode (one of `{"constant", "gaussian_noise"}`). Default: "constant".
            - *constant*: Pixels are filled with the same constant value.
            - *gaussian_noise*: Pixels are filled with random gaussian noise.
        fill_value: an integer represents of value to be filled inside the gridblock
            when `fill_mode="constant"`. Valid integer range [0 to 255]
        seed: Integer. Used to create a random seed.

    Usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    random_gridmask = keras_cv.layers.preprocessing.GridMask()
    augmented_images = random_gridmask(images)
    ```

    References:
        - https://arxiv.org/abs/2001.04086
    """

    def __init__(
        self,
        ratio_factor=(0, 0.5),
        rotation_factor=0.15,
        fill_mode="constant",
        fill_value=0.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.ratio_factor = preprocessing.parse_factor(
            ratio_factor, param_name="ratio_factor"
        )

        if isinstance(rotation_factor, core.FactorSampler):
            raise ValueError(
                "Currently `GridMask.rotation_factor` does not support the "
                "`FactorSampler` API.  This will be supported in the next Keras "
                "release. For now, please pass a float for the "
                "`rotation_factor` argument."
            )

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.rotation_factor = rotation_factor
        self.random_rotate = layers.RandomRotation(
            factor=rotation_factor,
            fill_mode="constant",
            fill_value=0.0,
            seed=seed,
        )
        self.auto_vectorize = False
        self._check_parameter_values()
        self.seed = seed

    def _check_parameter_values(self):
        fill_mode, fill_value = self.fill_mode, self.fill_value

        if fill_value not in range(0, 256):
            raise ValueError(
                f"fill_value should be in the range [0, 255]. Got {fill_value}"
            )

        if fill_mode not in ["constant", "gaussian_noise", "random"]:
            raise ValueError(
                '`fill_mode` should be "constant", '
                f'"gaussian_noise", or "random".  Got `fill_mode`={fill_mode}'
            )

    def get_random_transformation(
        self, image=None, label=None, bounding_boxes=None, **kwargs
    ):
        ratio = self.ratio_factor()

        # compute grid mask
        input_shape = tf.shape(image)
        mask = self._compute_grid_mask(input_shape, ratio=ratio)

        # convert mask to single-channel image
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, axis=-1)

        # randomly rotate mask
        mask = self.random_rotate(mask)

        # compute fill
        if self.fill_mode == "constant":
            fill_value = tf.fill(input_shape, self.fill_value)
        else:
            # gaussian noise
            fill_value = self._random_generator.random_normal(
                shape=input_shape, dtype=image.dtype
            )

        return mask, fill_value

    def _compute_grid_mask(self, input_shape, ratio):
        height = tf.cast(input_shape[0], tf.float32)
        width = tf.cast(input_shape[1], tf.float32)

        # mask side length
        input_diagonal_len = tf.sqrt(tf.square(width) + tf.square(height))
        mask_side_len = tf.math.ceil(input_diagonal_len)

        # grid unit size
        unit_size = self._random_generator.random_uniform(
            shape=(),
            minval=tf.math.minimum(height * 0.5, width * 0.3),
            maxval=tf.math.maximum(height * 0.5, width * 0.3) + 1,
            dtype=tf.float32,
        )
        rectangle_side_len = tf.cast((ratio) * unit_size, tf.float32)

        # sample x and y offset for grid units randomly between 0 and unit_size
        delta_x = self._random_generator.random_uniform(
            shape=(), minval=0.0, maxval=unit_size, dtype=tf.float32
        )
        delta_y = self._random_generator.random_uniform(
            shape=(), minval=0.0, maxval=unit_size, dtype=tf.float32
        )

        # grid size (number of diagonal units in grid)
        grid_size = mask_side_len // unit_size + 1
        grid_size_range = tf.range(1, grid_size + 1)

        # diagonal corner coordinates
        unit_size_range = grid_size_range * unit_size
        x1 = unit_size_range - delta_x
        x0 = x1 - rectangle_side_len
        y1 = unit_size_range - delta_y
        y0 = y1 - rectangle_side_len

        # compute grid coordinates
        x0, y0 = tf.meshgrid(x0, y0)
        x1, y1 = tf.meshgrid(x1, y1)

        # flatten mesh grid
        x0 = tf.reshape(x0, [-1])
        y0 = tf.reshape(y0, [-1])
        x1 = tf.reshape(x1, [-1])
        y1 = tf.reshape(y1, [-1])

        # convert coordinates to mask
        corners = tf.stack([x0, y0, x1, y1], axis=-1)
        mask_side_len = tf.cast(mask_side_len, tf.int32)
        rectangle_masks = fill_utils.corners_to_mask(
            corners, mask_shape=(mask_side_len, mask_side_len)
        )
        grid_mask = tf.reduce_any(rectangle_masks, axis=0)

        return grid_mask

    def augment_image(self, image, transformation=None, **kwargs):
        mask, fill_value = transformation
        input_shape = tf.shape(image)

        # center crop mask
        input_height = input_shape[0]
        input_width = input_shape[1]
        mask = _center_crop(mask, input_width, input_height)

        # convert back to boolean mask
        mask = tf.cast(mask, tf.bool)

        return tf.where(mask, fill_value, image)

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def get_config(self):
        config = {
            "ratio_factor": self.ratio_factor,
            "rotation_factor": self.rotation_factor,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
