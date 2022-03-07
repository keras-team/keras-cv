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


class RGBShift(layers.Layer):
    """RGBShift class randomly shift values for each channel of the 
    input RGB image. The expected images should be `(0-255)` pixel ranges.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `channels_last` format

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `channels_last` format

    Args:
        factor: A scalar or tuple or list of two upper and lower bound 
            number. If factor is a single value, the range will be 
            `(-factor, factor)`. The factor value can be float or integer; 
            for float the valid limits are `(-1.0, 1.0)` and for integer the 
            valid limits are `(-255, 255)`.
        seed: Integer. Used to create a random seed. Default: None.

    Call arguments: 
        images: Tensor representing images of shape
            `(..., height, width, channels)`, with dtype tf.float32 / tf.uint8, or,
            `(height, width, channels)`, with dtype tf.float32 / tf.uint8
        training: A boolean argument that determines whether the call should be run 
            in inference mode or training mode. Default: True.
   
    Usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    rgb_shift = keras_cv.layers.RGBShift(factor=(-2, 2))
    augmented_images = rgb_shift(images)
    ```
    """

    _FACTOR_VALIDATION_ERROR = (
        "The factor should be a scalar, "
        "a tuple or a list of two upper and lower "
        "bound values in the range `(-1.0, 1.0)` as float or `(-255, 255) as integer."
    )

    def __init__(self, factor, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.factor = self._set_factor_limit(factor)
        self.seed = seed

    def _set_factor_limit(self, factor):
        if isinstance(factor, (tuple, list)):
            if len(factor) != 2:
                raise ValueError(
                    self._FACTOR_VALIDATION_ERROR + f" Received: factor={factor}"
                )
            return self._check_factor_range(sorted(factor))
        elif isinstance(factor, (int, float)):
            factor = abs(factor)
            return self._check_factor_range([-factor, factor])
        else:
            raise ValueError(
                self._FACTOR_VALIDATION_ERROR + f" Received: factor={factor}"
            )

    def _check_factor_range(self, factor):
        if all(isinstance(each_elem, float) for each_elem in factor):
            if factor[0] < -1.0 or factor[1] > 1.0:
                raise ValueError(
                    self._FACTOR_VALIDATION_ERROR + f" Received: factor={factor}"
                )
            return factor
        elif all(isinstance(each_elem, int) for each_elem in factor):
            if factor[0] < -255 or factor[1] > 255:
                raise ValueError(
                    self._FACTOR_VALIDATION_ERROR + f" Received: factor={factor}"
                )
            return factor
        else:
            raise ValueError(
                "Both lower/upper bound values must have the same dtype. "
                f"Received: factor={factor} where type(factor[0]) is {type(factor[0])} "
                f"and type(factor[1]) is {type(factor[1])}"
            )

    def _get_random_uniform(self, factor_limit, rgb_delta_shape):
        if self.seed is not None:
            _rand_uniform = tf.random.stateless_uniform(
                shape=rgb_delta_shape,
                seed=[0, self.seed],
                minval=factor_limit[0],
                maxval=factor_limit[1],
            )
        else:
            _rand_uniform = tf.random.uniform(
                rgb_delta_shape,
                minval=factor_limit[0],
                maxval=factor_limit[1],
                dtype=tf.float32,
            )

        if all(isinstance(each_elem, float) for each_elem in factor_limit):
            _rand_uniform = _rand_uniform * 85.0

        return _rand_uniform

    def _rgb_shifting(self, images):
        rank = images.shape.rank
        original_dtype = images.dtype

        if rank == 3:
            rgb_delta_shape = (1, 1)
        elif rank == 4:
            # Keep only the batch dim. This will ensure to have same adjustment
            # with in one image, but different across the images.
            rgb_delta_shape = [tf.shape(images)[0], 1, 1]
        else:
            raise ValueError(
                f"Expect the input image to be rank 3 or 4. Got {images.shape}"
            )

        r_shift = self._get_random_uniform(self.factor, rgb_delta_shape)
        g_shift = self._get_random_uniform(self.factor, rgb_delta_shape)
        b_shift = self._get_random_uniform(self.factor, rgb_delta_shape)

        unstack_rgb = tf.unstack(tf.cast(images, dtype=tf.float32), axis=-1)
        shifted_rgb = tf.stack(
            [
                tf.add(unstack_rgb[0], r_shift),
                tf.add(unstack_rgb[1], g_shift),
                tf.add(unstack_rgb[2], b_shift),
            ],
            axis=-1,
        )
        shifted_rgb = tf.clip_by_value(shifted_rgb, 0.0, 255.0)

        return tf.cast(shifted_rgb, dtype=original_dtype)

    def call(self, images, training=True):
        if training:
            return self._rgb_shifting(images)
        else:
            return images

    def get_config(self):
        config = super().get_config()
        config.update({"factor": self.factor, "seed": self.seed})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
