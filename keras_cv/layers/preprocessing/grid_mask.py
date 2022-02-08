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
from tensorflow.keras import layers, backend
from tensorflow.python.keras.utils import layer_utils


class GridMask(layers.Layer):
    """GridMask class for grid-mask augmentation. The expected images should be [0-255] pixel ranges.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format

    Args:
        ratio: The ratio from grid masks to spacings.
            Float in range [0, 1]. Defaults to 0.5, which indicates that grid and spacing will be equal.
            In other word, higher value makes grid size smaller and equally spaced, and opposite.
        gridmask_rotation_factor:
            a float represented as fraction of 2 Pi, or a tuple of size 2 representing lower and upper
            bound for rotating clockwise and counter-clockwise. A positive values means rotating counter
            clock-wise, while a negative value means clock-wise. When represented as a single float, this
            value is used for both the upper and lower bound. For instance, factor=(-0.2, 0.3) results in
            an output rotation by a random amount in the range [-20% * 2pi, 30% * 2pi]. factor=0.2 results
            in an output rotating by a random amount in the range [-20% * 2pi, 20% * 2pi].

            The gridmask_rotation_factor will pass to tf.keras.layers.RandomRotation to apply random rotation
            on gridmask. A preprocessing layer which randomly rotates gridmask during training. Default to 0.1,
            which results in an output rotating by a random amount in the range [-10% * 2pi, 10% * 2pi].
        fill_mode: Pixels inside the gridblock are filled according to the given
            mode (one of `{"constant", "gaussian_noise"}`).
            - *constant*: Pixels are filled with the same constant value.
            - *gaussian_noise*: Pixels are filled with random gaussian noise.
        fill_value: an integer represents of value to be filled inside the gridblock
            when `fill_mode="constant"`. Valid integer range [1 to 255], where 1 for black towards 255 for white.
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
        ratio=0.6,
        gridmask_rotation_factor=0.1,
        fill_mode="constant",
        fill_value=1,
        seed=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        layer_utils.validate_string_arg(
            fill_mode,
            allowable_strings=["constant", "gaussian_noise"],
            layer_name="GridMask",
            arg_name="fill_mode",
            allow_none=False,
            allow_callables=False,
        )

        self.ratio = ratio
        self.gridmask_random_rotate = layers.RandomRotation(
            factor=gridmask_rotation_factor, seed=seed
        )
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.seed = seed

    @staticmethod
    def crop(mask, image_height, image_width):
        """crops in middle of mask and image corners."""
        ww = hh = tf.shape(mask)[0]
        mask = mask[
            (hh - image_height) // 2 : (hh - image_height) // 2 + image_height,
            (ww - image_width) // 2 : (ww - image_width) // 2 + image_width,
        ]
        return mask

    @tf.function
    def mask(self, image_height, image_width):
        """mask helper function for initializing grid mask of required size."""
        image_height = tf.cast(image_height, tf.float32)
        image_width = tf.cast(image_width, tf.float32)

        mask_w = mask_h = tf.cast(
            tf.math.maximum(image_height, image_width) * 2.0, tf.int32
        )

        if self.fill_mode == "constant":
            mask = tf.fill([mask_h, mask_w], self.fill_value)
        else:
            mask = tf.cast(tf.random.normal([mask_h, mask_w]), tf.int32)

        gridblock = tf.random.uniform(
            shape=[],
            minval=int(tf.math.minimum(image_height * 0.5, image_width * 0.3)),
            maxval=int(tf.math.maximum(image_height * 0.5, image_width * 0.3)) + 1,
            dtype=tf.int32,
            seed=self.seed,
        )

        if self.ratio == 1:
            length = tf.random.uniform(
                shape=[], minval=1, maxval=gridblock + 1, dtype=tf.int32, seed=self.seed
            )
        else:
            length = tf.cast(
                tf.math.minimum(
                    tf.math.maximum(
                        int(tf.cast(gridblock, tf.float32) * self.ratio + 0.5), 1
                    ),
                    gridblock - 1,
                ),
                tf.int32,
            )

        for _ in range(2):
            start_w = tf.random.uniform(
                shape=[], minval=0, maxval=gridblock + 1, dtype=tf.int32, seed=self.seed
            )

            for i in range(mask_w // gridblock):
                start = gridblock * i + start_w
                end = tf.math.minimum(start + length, mask_w)
                indices = tf.reshape(tf.range(start, end), [end - start, 1])

                if self.fill_mode == "constant":
                    updates = (
                        tf.zeros(shape=[end - start, mask_w], dtype=tf.int32)
                        * self.fill_value
                    )
                else:
                    updates = tf.ones(shape=[end - start, mask_w], dtype=tf.int32)

                mask = tf.tensor_scatter_nd_update(mask, indices, updates)
            mask = tf.transpose(mask)
        return mask

    @tf.function
    def _grid_mask(self, image):
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]
        grid = self.mask(image_height, image_width)
        grid = self.gridmask_random_rotate(grid[:, :, tf.newaxis])

        mask = tf.reshape(
            tf.cast(self.crop(grid, image_height, image_width), image.dtype),
            (image_height, image_width),
        )
        mask = tf.expand_dims(mask, -1) if image._rank() != mask._rank() else mask

        if self.fill_mode == "constant":
            return tf.where(mask < self.fill_value, image, mask)
        else:
            return mask * image

    def call(self, images, training=None):
        """call method for the GridMask layer.

        Args:
            images: Tensor representing images of shape
                [batch_size, width, height, channels], with dtype tf.float32, or,
                [width, height, channels], with dtype tf.float32
        Returns:
            images: augmented images, same shape as input.
        """

        if training is None:
            training = backend.learning_phase()

        if training:
            unbatched = images.shape.rank == 3

            # The transform op only accepts rank 4 inputs, so if we have an unbatched
            # image, we need to temporarily expand dims to a batch.
            if unbatched:
                images = tf.expand_dims(images, 0)

            # TODO: Make the batch operation vectorize.
            output = tf.map_fn(lambda image: self._grid_mask(image), images)

            if unbatched:
                output = tf.squeeze(output, 0)
            return output

        return images

    def get_config(self):
        config = {
            "ratio": self.ratio,
            "gridmask_rotation_factor": self.gridmask_rotation_factor,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
