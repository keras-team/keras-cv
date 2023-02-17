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
"""
# Setup/utils
"""
import time

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras import backend

from keras_cv.utils import bounding_box
from keras_cv.utils import fill_utils


def single_rectangle_mask(corners, mask_shape):
    """Computes masks of rectangles

    Args:
        corners: tensor of rectangle coordinates with shape (batch_size, 4) in
            corners format (x0, y0, x1, y1).
        mask_shape: a shape tuple as (width, height) indicating the output
            width and height of masks.

    Returns:
        boolean masks with shape (batch_size, width, height) where True values
            indicate positions within rectangle coordinates.
    """
    # add broadcasting axes
    corners = corners[..., tf.newaxis, tf.newaxis]

    # split coordinates
    x0 = corners[0]
    y0 = corners[1]
    x1 = corners[2]
    y1 = corners[3]

    # repeat height and width
    width, height = mask_shape
    x0_rep = tf.repeat(x0, height, axis=0)
    y0_rep = tf.repeat(y0, width, axis=1)
    x1_rep = tf.repeat(x1, height, axis=0)
    y1_rep = tf.repeat(y1, width, axis=1)

    # range grid
    range_row = tf.range(0, height, dtype=corners.dtype)
    range_col = tf.range(0, width, dtype=corners.dtype)
    range_row = range_row[:, tf.newaxis]
    range_col = range_col[tf.newaxis, :]

    # boolean masks
    mask_x0 = tf.less_equal(x0_rep, range_col)
    mask_y0 = tf.less_equal(y0_rep, range_row)
    mask_x1 = tf.less(range_col, x1_rep)
    mask_y1 = tf.less(range_row, y1_rep)

    masks = mask_x0 & mask_y0 & mask_x1 & mask_y1

    return masks


def fill_single_rectangle(
    image, centers_x, centers_y, widths, heights, fill_values
):
    """Fill rectangles with fill value into images.

    Args:
        images: Tensor of images to fill rectangles into.
        centers_x: Tensor of positions of the rectangle centers on the x-axis.
        centers_y: Tensor of positions of the rectangle centers on the y-axis.
        widths: Tensor of widths of the rectangles
        heights: Tensor of heights of the rectangles
        fill_values: Tensor with same shape as images to get rectangle fill from.
    Returns:
        images with filled rectangles.
    """
    images_shape = tf.shape(image)
    images_height = images_shape[0]
    images_width = images_shape[1]

    xywh = tf.stack([centers_x, centers_y, widths, heights], axis=0)
    xywh = tf.cast(xywh, tf.float32)
    corners = bounding_box.convert_to_corners(xywh, format="coco")

    mask_shape = (images_width, images_height)
    is_rectangle = single_rectangle_mask(corners, mask_shape)
    is_rectangle = tf.expand_dims(is_rectangle, -1)

    images = tf.where(is_rectangle, fill_values, image)
    return images


"""
# Layer Implementations
## Fully Vectorized
"""


class VectorizedRandomCutout(layers.Layer):
    def __init__(
        self,
        height_factor,
        width_factor,
        fill_mode="constant",
        fill_value=0.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.height_lower, self.height_upper = self._parse_bounds(height_factor)
        self.width_lower, self.width_upper = self._parse_bounds(width_factor)

        if fill_mode not in ["gaussian_noise", "constant"]:
            raise ValueError(
                '`fill_mode` should be "gaussian_noise" '
                f'or "constant".  Got `fill_mode`={fill_mode}'
            )

        if not isinstance(self.height_lower, type(self.height_upper)):
            raise ValueError(
                "`height_factor` must have lower bound and upper bound "
                "with same type, got {} and {}".format(
                    type(self.height_lower), type(self.height_upper)
                )
            )
        if not isinstance(self.width_lower, type(self.width_upper)):
            raise ValueError(
                "`width_factor` must have lower bound and upper bound "
                "with same type, got {} and {}".format(
                    type(self.width_lower), type(self.width_upper)
                )
            )

        if self.height_upper < self.height_lower:
            raise ValueError(
                "`height_factor` cannot have upper bound less than "
                "lower bound, got {}".format(height_factor)
            )
        self._height_is_float = isinstance(self.height_lower, float)
        if self._height_is_float:
            if not self.height_lower >= 0.0 or not self.height_upper <= 1.0:
                raise ValueError(
                    "`height_factor` must have values between [0, 1] "
                    "when is float, got {}".format(height_factor)
                )

        if self.width_upper < self.width_lower:
            raise ValueError(
                "`width_factor` cannot have upper bound less than "
                "lower bound, got {}".format(width_factor)
            )
        self._width_is_float = isinstance(self.width_lower, float)
        if self._width_is_float:
            if not self.width_lower >= 0.0 or not self.width_upper <= 1.0:
                raise ValueError(
                    "`width_factor` must have values between [0, 1] "
                    "when is float, got {}".format(width_factor)
                )

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.seed = seed

    def _parse_bounds(self, factor):
        if isinstance(factor, (tuple, list)):
            return factor[0], factor[1]
        else:
            return type(factor)(0), factor

    @tf.function(jit_compile=True)
    def call(self, inputs, training=True):
        if training is None:
            training = backend.learning_phase()

        augment = lambda: self._random_cutout(inputs)
        no_augment = lambda: inputs
        return tf.cond(tf.cast(training, tf.bool), augment, no_augment)

    def _random_cutout(self, inputs):
        """Apply random cutout."""
        center_x, center_y = self._compute_rectangle_position(inputs)
        rectangle_height, rectangle_width = self._compute_rectangle_size(inputs)
        rectangle_fill = self._compute_rectangle_fill(inputs)
        inputs = fill_utils.fill_rectangle(
            inputs,
            center_x,
            center_y,
            rectangle_width,
            rectangle_height,
            rectangle_fill,
        )
        return inputs

    def _compute_rectangle_position(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, image_height, image_width = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )
        center_x = tf.random.uniform(
            shape=[batch_size],
            minval=0,
            maxval=image_width,
            dtype=tf.int32,
            seed=self.seed,
        )
        center_y = tf.random.uniform(
            shape=[batch_size],
            minval=0,
            maxval=image_height,
            dtype=tf.int32,
            seed=self.seed,
        )
        return center_x, center_y

    def _compute_rectangle_size(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, image_height, image_width = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )
        height = tf.random.uniform(
            [batch_size],
            minval=self.height_lower,
            maxval=self.height_upper,
            dtype=tf.float32,
        )
        width = tf.random.uniform(
            [batch_size],
            minval=self.width_lower,
            maxval=self.width_upper,
            dtype=tf.float32,
        )

        if self._height_is_float:
            height = height * tf.cast(image_height, tf.float32)

        if self._width_is_float:
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


"""
## tf.map_fn
"""


class MapFnRandomCutout(layers.Layer):
    def __init__(
        self,
        height_factor,
        width_factor,
        fill_mode="constant",
        fill_value=0.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.height_lower, self.height_upper = self._parse_bounds(height_factor)
        self.width_lower, self.width_upper = self._parse_bounds(width_factor)

        if fill_mode not in ["gaussian_noise", "constant"]:
            raise ValueError(
                '`fill_mode` should be "gaussian_noise" '
                f'or "constant".  Got `fill_mode`={fill_mode}'
            )

        if not isinstance(self.height_lower, type(self.height_upper)):
            raise ValueError(
                "`height_factor` must have lower bound and upper bound "
                "with same type, got {} and {}".format(
                    type(self.height_lower), type(self.height_upper)
                )
            )
        if not isinstance(self.width_lower, type(self.width_upper)):
            raise ValueError(
                "`width_factor` must have lower bound and upper bound "
                "with same type, got {} and {}".format(
                    type(self.width_lower), type(self.width_upper)
                )
            )

        if self.height_upper < self.height_lower:
            raise ValueError(
                "`height_factor` cannot have upper bound less than "
                "lower bound, got {}".format(height_factor)
            )
        self._height_is_float = isinstance(self.height_lower, float)
        if self._height_is_float:
            if not self.height_lower >= 0.0 or not self.height_upper <= 1.0:
                raise ValueError(
                    "`height_factor` must have values between [0, 1] "
                    "when is float, got {}".format(height_factor)
                )

        if self.width_upper < self.width_lower:
            raise ValueError(
                "`width_factor` cannot have upper bound less than "
                "lower bound, got {}".format(width_factor)
            )
        self._width_is_float = isinstance(self.width_lower, float)
        if self._width_is_float:
            if not self.width_lower >= 0.0 or not self.width_upper <= 1.0:
                raise ValueError(
                    "`width_factor` must have values between [0, 1] "
                    "when is float, got {}".format(width_factor)
                )

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.seed = seed

    def _parse_bounds(self, factor):
        if isinstance(factor, (tuple, list)):
            return factor[0], factor[1]
        else:
            return type(factor)(0), factor

    @tf.function(jit_compile=True)
    def call(self, inputs, training=True):
        augment = lambda: tf.map_fn(self._random_cutout, inputs)
        no_augment = lambda: inputs
        return tf.cond(tf.cast(training, tf.bool), augment, no_augment)

    def _random_cutout(self, input):
        center_x, center_y = self._compute_rectangle_position(input)
        rectangle_height, rectangle_width = self._compute_rectangle_size(input)
        rectangle_fill = self._compute_rectangle_fill(input)
        input = fill_single_rectangle(
            input,
            center_x,
            center_y,
            rectangle_width,
            rectangle_height,
            rectangle_fill,
        )
        return input

    def _compute_rectangle_position(self, inputs):
        input_shape = tf.shape(inputs)
        image_height, image_width = (
            input_shape[0],
            input_shape[1],
        )
        center_x = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=image_width,
            dtype=tf.int32,
            seed=self.seed,
        )
        center_y = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=image_height,
            dtype=tf.int32,
            seed=self.seed,
        )
        return center_x, center_y

    def _compute_rectangle_size(self, inputs):
        input_shape = tf.shape(inputs)
        image_height, image_width = (
            input_shape[0],
            input_shape[1],
        )
        height = tf.random.uniform(
            [],
            minval=self.height_lower,
            maxval=self.height_upper,
            dtype=tf.float32,
        )
        width = tf.random.uniform(
            [],
            minval=self.width_lower,
            maxval=self.width_upper,
            dtype=tf.float32,
        )

        if self._height_is_float:
            height = height * tf.cast(image_height, tf.float32)

        if self._width_is_float:
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


"""
## tf.vectorized_map
"""


class VMapRandomCutout(layers.Layer):
    def __init__(
        self,
        height_factor,
        width_factor,
        fill_mode="constant",
        fill_value=0.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.height_lower, self.height_upper = self._parse_bounds(height_factor)
        self.width_lower, self.width_upper = self._parse_bounds(width_factor)

        if fill_mode not in ["gaussian_noise", "constant"]:
            raise ValueError(
                '`fill_mode` should be "gaussian_noise" '
                f'or "constant".  Got `fill_mode`={fill_mode}'
            )

        if not isinstance(self.height_lower, type(self.height_upper)):
            raise ValueError(
                "`height_factor` must have lower bound and upper bound "
                "with same type, got {} and {}".format(
                    type(self.height_lower), type(self.height_upper)
                )
            )
        if not isinstance(self.width_lower, type(self.width_upper)):
            raise ValueError(
                "`width_factor` must have lower bound and upper bound "
                "with same type, got {} and {}".format(
                    type(self.width_lower), type(self.width_upper)
                )
            )

        if self.height_upper < self.height_lower:
            raise ValueError(
                "`height_factor` cannot have upper bound less than "
                "lower bound, got {}".format(height_factor)
            )
        self._height_is_float = isinstance(self.height_lower, float)
        if self._height_is_float:
            if not self.height_lower >= 0.0 or not self.height_upper <= 1.0:
                raise ValueError(
                    "`height_factor` must have values between [0, 1] "
                    "when is float, got {}".format(height_factor)
                )

        if self.width_upper < self.width_lower:
            raise ValueError(
                "`width_factor` cannot have upper bound less than "
                "lower bound, got {}".format(width_factor)
            )
        self._width_is_float = isinstance(self.width_lower, float)
        if self._width_is_float:
            if not self.width_lower >= 0.0 or not self.width_upper <= 1.0:
                raise ValueError(
                    "`width_factor` must have values between [0, 1] "
                    "when is float, got {}".format(width_factor)
                )

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.seed = seed

    def _parse_bounds(self, factor):
        if isinstance(factor, (tuple, list)):
            return factor[0], factor[1]
        else:
            return type(factor)(0), factor

    @tf.function(jit_compile=True)
    def call(self, inputs, training=True):
        augment = lambda: tf.vectorized_map(self._random_cutout, inputs)
        no_augment = lambda: inputs
        return tf.cond(tf.cast(training, tf.bool), augment, no_augment)

    def _random_cutout(self, input):
        center_x, center_y = self._compute_rectangle_position(input)
        rectangle_height, rectangle_width = self._compute_rectangle_size(input)
        rectangle_fill = self._compute_rectangle_fill(input)
        input = fill_single_rectangle(
            input,
            center_x,
            center_y,
            rectangle_width,
            rectangle_height,
            rectangle_fill,
        )
        return input

    def _compute_rectangle_position(self, inputs):
        input_shape = tf.shape(inputs)
        image_height, image_width = (
            input_shape[0],
            input_shape[1],
        )
        center_x = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=image_width,
            dtype=tf.int32,
            seed=self.seed,
        )
        center_y = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=image_height,
            dtype=tf.int32,
            seed=self.seed,
        )
        return center_x, center_y

    def _compute_rectangle_size(self, inputs):
        input_shape = tf.shape(inputs)
        image_height, image_width = (
            input_shape[0],
            input_shape[1],
        )
        height = tf.random.uniform(
            [],
            minval=self.height_lower,
            maxval=self.height_upper,
            dtype=tf.float32,
        )
        width = tf.random.uniform(
            [],
            minval=self.width_lower,
            maxval=self.width_upper,
            dtype=tf.float32,
        )

        if self._height_is_float:
            height = height * tf.cast(image_height, tf.float32)

        if self._width_is_float:
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


"""
JIT COMPILED
# Layer Implementations
## Fully Vectorized
"""


class JITVectorizedRandomCutout(layers.Layer):
    def __init__(
        self,
        height_factor,
        width_factor,
        fill_mode="constant",
        fill_value=0.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.height_lower, self.height_upper = self._parse_bounds(height_factor)
        self.width_lower, self.width_upper = self._parse_bounds(width_factor)

        if fill_mode not in ["gaussian_noise", "constant"]:
            raise ValueError(
                '`fill_mode` should be "gaussian_noise" '
                f'or "constant".  Got `fill_mode`={fill_mode}'
            )

        if not isinstance(self.height_lower, type(self.height_upper)):
            raise ValueError(
                "`height_factor` must have lower bound and upper bound "
                "with same type, got {} and {}".format(
                    type(self.height_lower), type(self.height_upper)
                )
            )
        if not isinstance(self.width_lower, type(self.width_upper)):
            raise ValueError(
                "`width_factor` must have lower bound and upper bound "
                "with same type, got {} and {}".format(
                    type(self.width_lower), type(self.width_upper)
                )
            )

        if self.height_upper < self.height_lower:
            raise ValueError(
                "`height_factor` cannot have upper bound less than "
                "lower bound, got {}".format(height_factor)
            )
        self._height_is_float = isinstance(self.height_lower, float)
        if self._height_is_float:
            if not self.height_lower >= 0.0 or not self.height_upper <= 1.0:
                raise ValueError(
                    "`height_factor` must have values between [0, 1] "
                    "when is float, got {}".format(height_factor)
                )

        if self.width_upper < self.width_lower:
            raise ValueError(
                "`width_factor` cannot have upper bound less than "
                "lower bound, got {}".format(width_factor)
            )
        self._width_is_float = isinstance(self.width_lower, float)
        if self._width_is_float:
            if not self.width_lower >= 0.0 or not self.width_upper <= 1.0:
                raise ValueError(
                    "`width_factor` must have values between [0, 1] "
                    "when is float, got {}".format(width_factor)
                )

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.seed = seed

    def _parse_bounds(self, factor):
        if isinstance(factor, (tuple, list)):
            return factor[0], factor[1]
        else:
            return type(factor)(0), factor

    @tf.function(jit_compile=True)
    def call(self, inputs, training=True):
        if training is None:
            training = backend.learning_phase()

        augment = lambda: self._random_cutout(inputs)
        no_augment = lambda: inputs
        return tf.cond(tf.cast(training, tf.bool), augment, no_augment)

    def _random_cutout(self, inputs):
        """Apply random cutout."""
        center_x, center_y = self._compute_rectangle_position(inputs)
        rectangle_height, rectangle_width = self._compute_rectangle_size(inputs)
        rectangle_fill = self._compute_rectangle_fill(inputs)
        inputs = fill_utils.fill_rectangle(
            inputs,
            center_x,
            center_y,
            rectangle_width,
            rectangle_height,
            rectangle_fill,
        )
        return inputs

    def _compute_rectangle_position(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, image_height, image_width = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )
        center_x = tf.random.uniform(
            shape=[batch_size],
            minval=0,
            maxval=image_width,
            dtype=tf.int32,
            seed=self.seed,
        )
        center_y = tf.random.uniform(
            shape=[batch_size],
            minval=0,
            maxval=image_height,
            dtype=tf.int32,
            seed=self.seed,
        )
        return center_x, center_y

    def _compute_rectangle_size(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, image_height, image_width = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )
        height = tf.random.uniform(
            [batch_size],
            minval=self.height_lower,
            maxval=self.height_upper,
            dtype=tf.float32,
        )
        width = tf.random.uniform(
            [batch_size],
            minval=self.width_lower,
            maxval=self.width_upper,
            dtype=tf.float32,
        )

        if self._height_is_float:
            height = height * tf.cast(image_height, tf.float32)

        if self._width_is_float:
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


"""
## tf.map_fn
"""


class JITMapFnRandomCutout(layers.Layer):
    def __init__(
        self,
        height_factor,
        width_factor,
        fill_mode="constant",
        fill_value=0.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.height_lower, self.height_upper = self._parse_bounds(height_factor)
        self.width_lower, self.width_upper = self._parse_bounds(width_factor)

        if fill_mode not in ["gaussian_noise", "constant"]:
            raise ValueError(
                '`fill_mode` should be "gaussian_noise" '
                f'or "constant".  Got `fill_mode`={fill_mode}'
            )

        if not isinstance(self.height_lower, type(self.height_upper)):
            raise ValueError(
                "`height_factor` must have lower bound and upper bound "
                "with same type, got {} and {}".format(
                    type(self.height_lower), type(self.height_upper)
                )
            )
        if not isinstance(self.width_lower, type(self.width_upper)):
            raise ValueError(
                "`width_factor` must have lower bound and upper bound "
                "with same type, got {} and {}".format(
                    type(self.width_lower), type(self.width_upper)
                )
            )

        if self.height_upper < self.height_lower:
            raise ValueError(
                "`height_factor` cannot have upper bound less than "
                "lower bound, got {}".format(height_factor)
            )
        self._height_is_float = isinstance(self.height_lower, float)
        if self._height_is_float:
            if not self.height_lower >= 0.0 or not self.height_upper <= 1.0:
                raise ValueError(
                    "`height_factor` must have values between [0, 1] "
                    "when is float, got {}".format(height_factor)
                )

        if self.width_upper < self.width_lower:
            raise ValueError(
                "`width_factor` cannot have upper bound less than "
                "lower bound, got {}".format(width_factor)
            )
        self._width_is_float = isinstance(self.width_lower, float)
        if self._width_is_float:
            if not self.width_lower >= 0.0 or not self.width_upper <= 1.0:
                raise ValueError(
                    "`width_factor` must have values between [0, 1] "
                    "when is float, got {}".format(width_factor)
                )

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.seed = seed

    def _parse_bounds(self, factor):
        if isinstance(factor, (tuple, list)):
            return factor[0], factor[1]
        else:
            return type(factor)(0), factor

    @tf.function(jit_compile=True)
    def call(self, inputs, training=True):
        augment = lambda: tf.map_fn(self._random_cutout, inputs)
        no_augment = lambda: inputs
        return tf.cond(tf.cast(training, tf.bool), augment, no_augment)

    def _random_cutout(self, input):
        center_x, center_y = self._compute_rectangle_position(input)
        rectangle_height, rectangle_width = self._compute_rectangle_size(input)
        rectangle_fill = self._compute_rectangle_fill(input)
        input = fill_single_rectangle(
            input,
            center_x,
            center_y,
            rectangle_width,
            rectangle_height,
            rectangle_fill,
        )
        return input

    def _compute_rectangle_position(self, inputs):
        input_shape = tf.shape(inputs)
        image_height, image_width = (
            input_shape[0],
            input_shape[1],
        )
        center_x = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=image_width,
            dtype=tf.int32,
            seed=self.seed,
        )
        center_y = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=image_height,
            dtype=tf.int32,
            seed=self.seed,
        )
        return center_x, center_y

    def _compute_rectangle_size(self, inputs):
        input_shape = tf.shape(inputs)
        image_height, image_width = (
            input_shape[0],
            input_shape[1],
        )
        height = tf.random.uniform(
            [],
            minval=self.height_lower,
            maxval=self.height_upper,
            dtype=tf.float32,
        )
        width = tf.random.uniform(
            [],
            minval=self.width_lower,
            maxval=self.width_upper,
            dtype=tf.float32,
        )

        if self._height_is_float:
            height = height * tf.cast(image_height, tf.float32)

        if self._width_is_float:
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


"""
## tf.vectorized_map
"""


class JITVMapRandomCutout(layers.Layer):
    def __init__(
        self,
        height_factor,
        width_factor,
        fill_mode="constant",
        fill_value=0.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.height_lower, self.height_upper = self._parse_bounds(height_factor)
        self.width_lower, self.width_upper = self._parse_bounds(width_factor)

        if fill_mode not in ["gaussian_noise", "constant"]:
            raise ValueError(
                '`fill_mode` should be "gaussian_noise" '
                f'or "constant".  Got `fill_mode`={fill_mode}'
            )

        if not isinstance(self.height_lower, type(self.height_upper)):
            raise ValueError(
                "`height_factor` must have lower bound and upper bound "
                "with same type, got {} and {}".format(
                    type(self.height_lower), type(self.height_upper)
                )
            )
        if not isinstance(self.width_lower, type(self.width_upper)):
            raise ValueError(
                "`width_factor` must have lower bound and upper bound "
                "with same type, got {} and {}".format(
                    type(self.width_lower), type(self.width_upper)
                )
            )

        if self.height_upper < self.height_lower:
            raise ValueError(
                "`height_factor` cannot have upper bound less than "
                "lower bound, got {}".format(height_factor)
            )
        self._height_is_float = isinstance(self.height_lower, float)
        if self._height_is_float:
            if not self.height_lower >= 0.0 or not self.height_upper <= 1.0:
                raise ValueError(
                    "`height_factor` must have values between [0, 1] "
                    "when is float, got {}".format(height_factor)
                )

        if self.width_upper < self.width_lower:
            raise ValueError(
                "`width_factor` cannot have upper bound less than "
                "lower bound, got {}".format(width_factor)
            )
        self._width_is_float = isinstance(self.width_lower, float)
        if self._width_is_float:
            if not self.width_lower >= 0.0 or not self.width_upper <= 1.0:
                raise ValueError(
                    "`width_factor` must have values between [0, 1] "
                    "when is float, got {}".format(width_factor)
                )

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.seed = seed

    def _parse_bounds(self, factor):
        if isinstance(factor, (tuple, list)):
            return factor[0], factor[1]
        else:
            return type(factor)(0), factor

    @tf.function(jit_compile=True)
    def call(self, inputs, training=True):
        augment = lambda: tf.vectorized_map(self._random_cutout, inputs)
        no_augment = lambda: inputs
        return tf.cond(tf.cast(training, tf.bool), augment, no_augment)

    def _random_cutout(self, input):
        center_x, center_y = self._compute_rectangle_position(input)
        rectangle_height, rectangle_width = self._compute_rectangle_size(input)
        rectangle_fill = self._compute_rectangle_fill(input)
        input = fill_single_rectangle(
            input,
            center_x,
            center_y,
            rectangle_width,
            rectangle_height,
            rectangle_fill,
        )
        return input

    def _compute_rectangle_position(self, inputs):
        input_shape = tf.shape(inputs)
        image_height, image_width = (
            input_shape[0],
            input_shape[1],
        )
        center_x = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=image_width,
            dtype=tf.int32,
            seed=self.seed,
        )
        center_y = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=image_height,
            dtype=tf.int32,
            seed=self.seed,
        )
        return center_x, center_y

    def _compute_rectangle_size(self, inputs):
        input_shape = tf.shape(inputs)
        image_height, image_width = (
            input_shape[0],
            input_shape[1],
        )
        height = tf.random.uniform(
            [],
            minval=self.height_lower,
            maxval=self.height_upper,
            dtype=tf.float32,
        )
        width = tf.random.uniform(
            [],
            minval=self.width_lower,
            maxval=self.width_upper,
            dtype=tf.float32,
        )

        if self._height_is_float:
            height = height * tf.cast(image_height, tf.float32)

        if self._width_is_float:
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


"""
# Benchmarking
"""
(x_train, _), _ = keras.datasets.cifar10.load_data()
x_train = x_train.astype(float)

x_train.shape


images = []

num_images = [1000, 2000, 5000, 10000, 25000, 37500, 50000]

results = {}

for aug in [
    VectorizedRandomCutout,
    VMapRandomCutout,
    MapFnRandomCutout,
    JITVectorizedRandomCutout,
    JITVMapRandomCutout,
    JITMapFnRandomCutout,
]:
    c = aug.__name__
    layer = aug(0.2, 0.2)
    runtimes = []
    print(f"Timing {c}")

    for n_images in num_images:
        # warmup
        layer(x_train[:n_images])

        t0 = time.time()
        r1 = layer(x_train[:n_images])
        t1 = time.time()
        runtimes.append(t1 - t0)
        print(f"Runtime for {c}, n_images={n_images}: {t1-t0}")

    results[c] = runtimes

plt.figure()
for key in results:
    plt.plot(num_images, results[key], label=key)
    plt.xlabel("Number images")

plt.ylabel("Runtime (seconds)")
plt.legend()
plt.show()

"""
# Sanity check
all of these should have comparable outputs
"""

images = []
for aug in [VectorizedRandomCutout, VMapRandomCutout, MapFnRandomCutout]:
    layer = aug(0.5, 0.5)
    images.append(layer(x_train[:3]))
images = [y for x in images for y in x]


plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.axis("off")
plt.show()

"""
# Extra notes

## Warnings
it would be really annoying as a user to use an official keras_cv component and get
warned that "RandomUniform" or "RandomUniformInt" inside pfor may not get the same
output.
"""
