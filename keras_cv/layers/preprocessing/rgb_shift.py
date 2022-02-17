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

import numpy as np 
import tensorflow as tf
from tensorflow.keras import layers, backend


class RGBShift(layers.Layer):
    """RGBShift class randomly shift values for each channel of the input RGB image. The expected images 
    should be [0-255] pixel ranges.

   
    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format

    Args:
        r_shift_limit: A scalar or tuple or list of two upper and lower bound number. It will change r-channel by
            adding with the r-channel. If r_shift_limit is a single integer, the range will be (-r_shift_limit, r_shift_limit).
            Default: (-10, 10).
        g_shift_limit: A scalar or tuple or list of two upper and lower bound number. It will change g-channel by
            adding with the r-channel. If r_shift_limit is a single integer, the range will be (-g_shift_limit, g_shift_limit).
            Default: (-10, 10).
        b_shift_limit: A scalar or tuple or list of two upper and lower bound number. It will change b-channel by
            adding with the r-channel. If r_shift_limit is a single integer, the range will be (-b_shift_limit, b_shift_limit).
            Default: (-10, 10).
        seed: Integer. Used to create a random seed. Default: None.

    Usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    rgbshift = keras_cv.layers.preprocessing.RGBShift()
    augmented_images = rgbshift(images)
    ```
    """

    def __init__(
        self,
        r_shift_limit=10,
        g_shift_limit=10,
        b_shift_limit=10,
        seed=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._r_shift_limit = self._set_shift_limit(r_shift_limit, channel='r_shift_limit')
        self._g_shift_limit = self._set_shift_limit(g_shift_limit, channel='g_shift_limit')
        self._b_shift_limit = self._set_shift_limit(b_shift_limit, channel='b_shift_limit')
        self.seed = seed

    def _set_shift_limit(self, shift_limit, channel=''):
        if isinstance(shift_limit, (tuple, list)):
            if len(shift_limit) != 2: 
                raise ValueError(f'The {channel} should be scalar, tuple or list of two upper and lower \
                    bound number. Got {shift_limit}')
            return self._check_scale_range(sorted(shift_limit))
        elif isinstance(shift_limit, (int, float)):
            shift_limit = abs(shift_limit)
            return self._check_scale_range(sorted(shift_limit))
        else:
            raise ValueError(f'The {channel} should be scalar, tuple or list of two upper and lower bound\
                 number. Got {shift_limit}')
        
    @staticmethod
    def _check_scale_range(shift_limit):
        if all(isinstance(each_elem, float) for each_elem in shift_limit):
            if shift_limit[0] < -1.0 or shift_limit[1] > 1.0:
                raise ValueError(f"Got {shift_limit}")
            return shift_limit
        elif all(isinstance(each_elem, int) for each_elem in shift_limit):
            if shift_limit[0] < -255 or shift_limit[1] > 255:
                raise ValueError(f"Got {shift_limit}")
            return shift_limit
        else:
            raise ValueError(f'Both bound must be same dtype. Got {shift_limit}')

    def _get_random_uniform(self, shift_limit, rgb_delta_shape):
            if self.seed is not None:
                _rand_uniform = tf.random.stateless_uniform(
                    shape=rgb_delta_shape,
                    seed=[0, self._seed],
                    minval=shift_limit[0],
                    maxval=shift_limit[1],
                )
            else:
                _rand_uniform = tf.random.uniform(rgb_delta_shape, 
                                                minval=shift_limit[0], 
                                                maxval=shift_limit[1], 
                                                dtype=tf.float32)
            

            if all(isinstance(each_elem, float) for each_elem in shift_limit):
                _rand_uniform = _rand_uniform * 85.0

            return _rand_uniform

    def _rgb_shifting(self, images):
        rank = images.shape.rank
        if rank == 3:
            rgb_delta_shape = (1, 1, 1)
        elif rank == 4:
            # Keep only the batch dim. This will ensure to have same adjustment
            # with in one image, but different across the images.
            rgb_delta_shape = [tf.shape(images)[0], 1, 1, 1]
        else:
            raise ValueError(
                f"Expect the input image to be rank 3 or 4. Got {images.shape}"
            )

        r_shift = self._get_random_uniform(self.r_shift_limit, rgb_delta_shape)   
        g_shift = self._get_random_uniform(self.g_shift_limit, rgb_delta_shape)
        b_shift = self._get_random_uniform(self.b_shift_limit, rgb_delta_shape)

        unstack_rgb = tf.unstack(images, axis=-1)
        shifted_rgb = tf.stack([unstack_rgb[0] + r_shift, 
                                unstack_rgb[1] + g_shift, 
                                unstack_rgb[2] + b_shift],  axis=-1)
        
        shifted_rgb = tf.clip_by_value(shifted_rgb, 0, 255)
        return tf.cast(shifted_rgb, images.dtype)

    def call(self, images, training=None):
        """call method for the RGBShift layer.

        Args:
            images: Tensor representing images of shape
                [batch_size, width, height, channels], with dtype tf.float32 / tf.uint8, or,
                [width, height, channels], with dtype tf.float32 / tf.uint8
        Returns:
            images: augmented images, same shape as input.
        """
        if training is None:
            training = backend.learning_phase()

        if not training:
            return images
        else:
            unbatched = images.shape.rank == 3
            # The transform op only accepts rank 4 inputs, so if we have an unbatched
            # image, we need to temporarily expand dims to a batch.
            if unbatched:
                images = tf.expand_dims(images, axis=0)

            # TODO: Check if input tensor is RGB or not. 
            # TODO: Support tf.uint8

            # TODO: Make the batch operation vectorize.
            output = tf.map_fn(lambda image: self._rgb_shifting(image), images)

            if unbatched:
                output = tf.squeeze(output, axis=0)
            return output

    def get_config(self):
        config = {
            "r_shift_limit": self._r_shift_limit,
            "g_shift_limit": self._g_shift_limit,
            "b_shift_limit": self._b_shift_limit,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape