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
            In orther word, higher value makes grid size smaller and equally spaced, and opposite.
        gridmask_rotation_factor:
            a float represented as fraction of 2 Pi, or a tuple of size 2 representing lower and upper 
            bound for rotating clockwise and counter-clockwise. A positive values means rotating counter
             clock-wise, while a negative value means clock-wise. When represented as a single float, this 
             value is used for both the upper and lower bound. For instance, factor=(-0.2, 0.3) results in 
             an output rotation by a random amount in the range [-20% * 2pi, 30% * 2pi]. factor=0.2 results 
             in an output rotating by a random amount in the range [-20% * 2pi, 20% * 2pi].

             The gridmask_rotation_factor will pass to layers.RandomRotation to apply random rotation on
            gridmask. A preprocessing layer which randomly rotates gridmask during training.
        seed:
            Integer. Used to create a random seed.

    Sample usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    random_gridmask = keras_cv.layers.preprocessing.GridMask(0.5)
    augmented_images = random_gridmask(images)
    ```
    """

    def __init__(self, ratio=0.6, gridmask_rotation_factor=0.1, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.gridmask_random_rotate = layers.RandomRotation(
            factor=gridmask_rotation_factor, seed=seed
        )
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

        mask = tf.zeros(shape=[mask_h, mask_w], dtype=tf.int32)
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
                updates = (
                    tf.ones(shape=[end - start, mask_w], dtype=tf.int32)
                    * self.fill_value
                )
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
        return image * mask

    def call(self, images, training=True):
        """Masks input image tensor with random grid mask."""
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
        config = {"ratio": self.ratio, "seed": self.seed}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
