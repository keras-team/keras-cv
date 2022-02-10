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
from tensorflow.python.keras.utils import layer_utils


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
            return sorted(shift_limit)
        elif isinstance(shift_limit, (int, float)):
            shift_limit = abs(shift_limit)
            return [-shift_limit, shift_limit]
        else:
            raise ValueError(f'The {channel} should be scalar, tuple or list of two upper and lower bound\
                 number. Got {shift_limit}')

    @tf.function
    def _rgb_shifting(self, image):
        r_shift = tf.random.uniform([], 
                                    minval=self._r_shift_limit[0], 
                                    maxval=self._r_shift_limit[1], 
                                    dtype=tf.float32, seed=self.seed)
        g_shift = tf.random.uniform([], 
                                    minval=self._g_shift_limit[0], 
                                    maxval=self._g_shift_limit[1], 
                                    dtype=tf.float32, seed=self.seed)
        b_shift = tf.random.uniform([], 
                                    minval=self._b_shift_limit[0],
                                    maxval=self._b_shift_limit[1], 
                                    dtype=tf.float32, seed=self.seed)
        
        unstack_rgb = tf.unstack(image, axis=-1)
        shifted_rgb = tf.stack([unstack_rgb[0] + r_shift, 
                                unstack_rgb[1] + g_shift, 
                                unstack_rgb[2] + b_shift],  axis=-1)
        shifted_rgb = tf.clip_by_value(shifted_rgb, 0, 255)
        return tf.cast(shifted_rgb, image.dtype)

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