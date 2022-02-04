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
    
    Args:
        ratio: The ratio from grit masks to spacings. 
            Float in range [0, 1]. Defaults to 0.5, which indicates that grid and spacing will be equal.
            In orther word, higher value makes grid size smaller and equally spaced, and opposite. 
        rate: Float between 0 and 1. The probability of augmenting an input.
            Defaults to 0.5.

    Sample usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    random_gridmask = keras_cv.layers.preprocessing.GridMask(0.5, 0.5, rate=1.0)
    augmented_images = random_gridmask(images)
    ```
    """
    
    def __init__(
        self,
        ratio=0.6,
        rate=0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.fill_value = 1 # TODO: make it adaptive, i.e. 'constant' or 'gaussian_noise'
        self.rate = rate
        # TODO: set seed for deterministic result

    @staticmethod
    def crop(mask, image_height, image_width):
        '''crops in middle of mask and image corners.'''
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
            dtype=tf.int32
        )

        if self.ratio == 1:
            length = tf.random.uniform(
                shape=[], minval=1, maxval=gridblock + 1, dtype=tf.int32
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
                shape=[], minval=0, maxval=gridblock + 1, dtype=tf.int32
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
    def _grid_mask(self, image, training):
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]
        grid = self.mask(image_height, image_width)
        # TODO: rnadomly rotate the grid, i.e. tf.image.rot90

        mask = tf.reshape(
            tf.cast(self.crop(grid, image_height, image_width), image.dtype),
            (image_height, image_width),
        )
        mask = tf.expand_dims(mask, -1) if image._rank() != mask._rank() else mask

        rate_cond = tf.less(
            tf.random.uniform(shape=[], minval=0, maxval=1.0), self.rate
        )
        augment_cond = tf.logical_and(rate_cond, training)
        return tf.cond(augment_cond, lambda: image * mask, lambda: image)

    def call(self, images, training=True):
        """Masks input image tensor with random grid mask."""
        if training is None:
            training = backend.learning_phase()
        
        # TODO: Make the batch operation vectorize.
        return tf.map_fn(lambda image: self._grid_mask(image, training), images)

    def get_config(self):
        config = {
            "ratio": self.ratio,
            "rate": self.rate,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
