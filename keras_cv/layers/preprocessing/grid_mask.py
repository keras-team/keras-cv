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
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.python.keras.utils import layer_utils

from keras_cv.utils import fill_utils


class GridMask(layers.Layer):
    """GridMask class for grid-mask augmentation.


    Input shape:
        Int or float tensor with values in the range [0, 255].
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format

    Args:
        ratio: The ratio from grid masks to spacings. Higher values make the grid
            size smaller, and large values make the grid mask large.
            Float in range [0, 1), defaults to 0.5, which indicates that grid and
            spacing will be of equal size.
            String value "random" will choose a random scale at each call.
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
        seed:
            Integer. Used to create a random seed.

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
        ratio="random",
        rotation_factor=0.15,
        fill_mode="constant",
        fill_value=0.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ratio = ratio
        if isinstance(ratio, str):
            self.ratio = ratio.lower()
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.random_rotate = layers.RandomRotation(
            factor=rotation_factor, fill_mode="constant", fill_value=0.0, seed=seed
        )
        self.seed = seed

        self._check_parameter_values()

    def _check_parameter_values(self):
        ratio, fill_mode, fill_value = self.ratio, self.fill_mode, self.fill_value
        ratio_error = (
            "ratio should be in the range [0.0, 1.0] or the string "
            f"'random'. Got {ratio}"
        )
        if isinstance(ratio, float):
            if ratio < 0 or ratio > 1:
                raise ValueError(ratio_error)
        elif isinstance(ratio, str) and ratio != "random":
            raise ValueError(ratio_error)

        if not isinstance(ratio, (str, float)):
            raise ValueError(ratio_error)

        if fill_value not in range(0, 256):
            raise ValueError(
                f"fill_value should be in the range [0, 255]. Got {fill_value}"
            )

        layer_utils.validate_string_arg(
            fill_mode,
            allowable_strings=["constant", "gaussian_noise"],
            layer_name="GridMask",
            arg_name="fill_mode",
            allow_none=False,
            allow_callables=False,
        )

    def _compute_grid_masks(self, inputs):
        """Computes grid masks"""
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        height = tf.cast(input_shape[1], tf.float32)
        width = tf.cast(input_shape[2], tf.float32)

        # masks side length
        squared_w = tf.square(width)
        squared_h = tf.square(height)
        mask_side_length = tf.math.ceil(tf.sqrt(squared_w + squared_h))
        mask_side_length = tf.cast(mask_side_length, tf.int32)

        # grid unit sizes
        unit_sizes = tf.random.uniform(
            shape=[batch_size],
            minval=tf.math.minimum(height * 0.5, width * 0.3),
            maxval=tf.math.maximum(height * 0.5, width * 0.3) + 1,
        )
        if self.ratio == "random":
            ratio = tf.random.uniform(
                shape=[batch_size], minval=0, maxval=1, dtype=tf.float32, seed=self.seed
            )
        else:
            ratio = self.ratio
        rectangle_side_length = tf.cast((1 - ratio) * unit_sizes, tf.int32)

        # x and y offsets for grid units
        delta_x = tf.random.uniform([batch_size], minval=0, maxval=1, dtype=tf.float32)
        delta_y = tf.random.uniform([batch_size], minval=0, maxval=1, dtype=tf.float32)
        delta_x = tf.cast(delta_x * unit_sizes, tf.int32)
        delta_y = tf.cast(delta_y * unit_sizes, tf.int32)

        # grid size (number of diagonal units per grid)
        unit_sizes = tf.cast(unit_sizes, tf.int32)
        grid_sizes = mask_side_length // unit_sizes + 1
        max_grid_size = tf.reduce_max(grid_sizes)

        # grid size range per image
        grid_size_range = tf.range(1, max_grid_size + 1)
        grid_size_range = tf.tile(tf.expand_dims(grid_size_range, 0), [batch_size, 1])

        # make broadcastable to grid size ranges
        delta_x = tf.expand_dims(delta_x, 1)
        delta_y = tf.expand_dims(delta_y, 1)
        unit_sizes = tf.expand_dims(unit_sizes, 1)
        rectangle_side_length = tf.expand_dims(rectangle_side_length, 1)

        # diagonal corner coordinates
        d_range = grid_size_range * unit_sizes
        x1 = d_range - delta_x
        x0 = x1 - rectangle_side_length
        y1 = d_range - delta_y
        y0 = y1 - rectangle_side_length

        # mask coordinates by grid ranges
        d_range_mask = tf.sequence_mask(
            lengths=grid_sizes, maxlen=max_grid_size, dtype=tf.int32
        )
        x1 = x1 * d_range_mask
        x0 = x0 * d_range_mask
        y1 = y1 * d_range_mask
        y0 = y0 * d_range_mask

        # mesh grid of diagonal top left corner coordinates for each image
        x0 = tf.tile(tf.expand_dims(x0, 1), [1, max_grid_size, 1])
        y0 = tf.tile(tf.expand_dims(y0, 1), [1, max_grid_size, 1])
        y0 = tf.transpose(y0, [0, 2, 1])

        # mesh grid of diagonal bottom right corner coordinates for each image
        x1 = tf.tile(tf.expand_dims(x1, 1), [1, max_grid_size, 1])
        y1 = tf.tile(tf.expand_dims(y1, 1), [1, max_grid_size, 1])
        y1 = tf.transpose(y1, [0, 2, 1])

        # flatten mesh grids
        x0 = tf.reshape(x0, [-1, max_grid_size])
        y0 = tf.reshape(y0, [-1, max_grid_size])
        x1 = tf.reshape(x1, [-1, max_grid_size])
        y1 = tf.reshape(y1, [-1, max_grid_size])

        # combine coordinates to (x0, y0, x1, y1)
        # with shape (num_rectangles_in_batch, 4)
        corners0 = tf.stack([x0, y0], axis=-1)
        corners1 = tf.stack([x1, y1], axis=-1)
        corners0 = tf.reshape(corners0, [-1, 2])
        corners1 = tf.reshape(corners1, [-1, 2])
        corners = tf.concat([corners0, corners1], axis=1)

        # make mask for each rectangle
        masks = fill_utils.rectangle_masks(
            corners, (mask_side_length, mask_side_length)
        )

        # reshape masks into shape
        # (batch_size, rectangles_per_image, mask_height, mask_width)
        masks = tf.reshape(
            masks,
            [-1, max_grid_size * max_grid_size, mask_side_length, mask_side_length],
        )

        # combine rectangle masks per image
        masks = tf.reduce_any(masks, axis=1)

        return masks

    def _center_crop(self, masks, width, height):
        masks_shape = tf.shape(masks)
        h_diff = masks_shape[1] - height
        w_diff = masks_shape[2] - width

        h_start = tf.cast(h_diff / 2, tf.int32)
        w_start = tf.cast(w_diff / 2, tf.int32)
        return tf.image.crop_to_bounding_box(masks, h_start, w_start, height, width)

    def _grid_mask(self, images):
        # compute grid masks
        masks = self._compute_grid_masks(images)

        # convert masks to single-channel images
        masks = tf.cast(masks, tf.uint8)
        masks = tf.expand_dims(masks, axis=-1)

        # randomly rotate masks
        masks = self.random_rotate(masks)

        # center crop masks
        input_shape = tf.shape(images)
        input_height = input_shape[1]
        input_width = input_shape[2]
        masks = self._center_crop(masks, input_width, input_height)

        # convert back to boolean mask
        masks = tf.cast(masks, tf.bool)

        if self.fill_mode == "constant":
            fill_value = tf.fill(input_shape, self.fill_value)
        else:
            # gaussian noise
            fill_value = tf.random.normal(input_shape)

        return tf.where(masks, fill_value, images)

    def call(self, images, training=True):
        """call method for the GridMask layer.

        Args:
            images: Tensor representing images with shape
                [batch_size, width, height, channels] or [width, height, channels]
                of type int or float.  Values should be in the range [0, 255].
        Returns:
            images: augmented images, same shape as input.
        """

        if training is None:
            training = backend.learning_phase()

        if training:
            images = self._grid_mask(images)

        return images

    def get_config(self):
        config = {
            "ratio": self.ratio,
            "rotation_factor": self.rotation_factor,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
