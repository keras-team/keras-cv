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
        fill_value=0,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ratio = ratio
        if isinstance(ratio, str):
            self.ratio = ratio.lower()
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.random_rotate = layers.RandomRotation(factor=rotation_factor, seed=seed)
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

        if fill_mode not in ["constant", "gaussian_noise", "random"]:
            raise ValueError(
                '`fill_mode` should be "constant", '
                f'"gaussian_noise", or "random".  Got `fill_mode`={fill_mode}'
            )

    @staticmethod
    def _crop(mask, image_height, image_width):
        """crops in middle of mask and image corners."""
        mask_width = mask_height = tf.shape(mask)[0]
        mask = mask[
            (mask_height - image_height) // 2 : (mask_height - image_height) // 2
            + image_height,
            (mask_width - image_width) // 2 : (mask_width - image_width) // 2
            + image_width,
        ]
        return mask

    @tf.function
    def _compute_mask(self, image_height, image_width):
        """mask helper function for initializing grid mask of required size."""
        image_height = tf.cast(image_height, dtype=tf.float32)
        image_width = tf.cast(image_width, dtype=tf.float32)

        mask_width = mask_height = tf.cast(
            tf.math.maximum(image_height, image_width) * 2.0, dtype=tf.int32
        )

        if self.fill_mode == "constant":
            mask = tf.fill([mask_height, mask_width], value=-1)
        elif self.fill_mode == "gaussian_noise":
            mask = tf.cast(tf.random.normal([mask_height, mask_width]), dtype=tf.int32)
        else:
            raise ValueError(
                "Unsupported fill_mode.  `fill_mode` should be 'constant' or "
                "'gaussian_noise'."
            )

        gridblock = tf.random.uniform(
            shape=[],
            minval=int(tf.math.minimum(image_height * 0.5, image_width * 0.3)),
            maxval=int(tf.math.maximum(image_height * 0.5, image_width * 0.3)) + 1,
            dtype=tf.int32,
            seed=self.seed,
        )

        if self.ratio == "random":
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
            start_x = tf.random.uniform(
                shape=[], minval=0, maxval=gridblock + 1, dtype=tf.int32, seed=self.seed
            )

            for i in range(mask_width // gridblock):
                start = gridblock * i + start_x
                end = tf.math.minimum(start + length, mask_width)
                indices = tf.reshape(tf.range(start, end), [end - start, 1])
                updates = tf.fill([end - start, mask_width], value=self.fill_value)
                mask = tf.tensor_scatter_nd_update(mask, indices, updates)
            mask = tf.transpose(mask)

        return tf.equal(mask, self.fill_value)

    @tf.function
    def _grid_mask(self, image):
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        grid = self._compute_mask(image_height, image_width)
        grid = self.random_rotate(tf.cast(grid[:, :, tf.newaxis], tf.float32))

        mask = tf.reshape(
            tf.cast(self._crop(grid, image_height, image_width), dtype=image.dtype),
            (image_height, image_width),
        )
        mask = tf.expand_dims(mask, -1) if image._rank() != mask._rank() else mask

        if self.fill_mode == "constant":
            return tf.where(tf.cast(mask, tf.bool), image, self.fill_value)
        else:
            return mask * image

    def _augment_images(self, images):
        unbatched = images.shape.rank == 3

        # The transform op only accepts rank 4 inputs, so if we have an unbatched
        # image, we need to temporarily expand dims to a batch.
        if unbatched:
            images = tf.expand_dims(images, axis=0)

        # TODO: Make the batch operation vectorize.
        output = tf.map_fn(lambda image: self._grid_mask(image), images)

        if unbatched:
            output = tf.squeeze(output, axis=0)
        return output

    def call(self, images, training=None):
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

        if not training:
            return images
        return self._augment_images(images)

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
