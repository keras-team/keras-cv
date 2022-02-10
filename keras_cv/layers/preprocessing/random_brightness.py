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

_SCALE_VALIDATION_ERROR = ('The `scale` should be number or a list of two numbers '
                           'that ranged between [-1.0, 1.0]. ')


class RandomBrightness(tf.keras.layers.Layer):
    """Randomly adjust brightness for the a RGB image.

    This layer will randomly increase/reduce the brightness for the input RGB image.
    During inference time, the output will be identical to input. Call the layer with
    training=True to adjust brightness of the input.

    Note that same brightness adjustment will be apply to all the images in the same
    batch.

    Args:
        scale: Float or a list/tuple of 2 floats between -1.0 and 1.0. The scale is
            used to determine the lower bound and upper bound of the brightness
            adjustment. A float value will be choose randomly between the limits.
            When -1 is chosen, the output image will be black, and when 1.0 is
            chosen, the image will be fully white. When only one float is provided,
            eg, 0.2, then -0.2 will be used for lower bound and 0.2 will be used for
             upper bound.
        seed: integer, for fixed RNG behavior.

    Inputs:
        3D (HWC) or 4D (NHWC) tensor, with float or int dtype. The value should be
        ranged between [0, 255].

    Output:
        3D (HWC) or 4D (NHWC) tensor with brightness adjusted based on the `scale`.
        The output will have same dtypes as the input image.

    Sample usage:
    ```
      random_bright = keras_cv.layers.RandomBrightness(scale=0.2)
      # An image with shape [2, 2, 3]
      image = [[[1, 2, 3], [4 ,5 ,6]],
               [[7, 8, 9], [10, 11, 12]]]
      # Assume we randomly select the scale to be 0.1, then it will apply 0.1 * 255 to
      # all the channel
      output = random_bright(image, training=True)
      # output will be int64 with 25.5 added to each channel and round down.
      tf.Tensor(
        [[[26 27 28]
          [29 30 31]]
         [[32 33 34]
          [35 36 37]]], shape=(2, 2, 3), dtype=int64)
    ```
    """

    def __init__(self, scale, seed=None, **kwargs):
        super().__init__(**kwargs)
        self._set_scale(scale)
        self._seed = seed

    def _set_scale(self, scale):
        if isinstance(scale, (tuple, list)):
            if len(scale) != 2:
                raise ValueError(_SCALE_VALIDATION_ERROR + f'Got {scale}')
            self._check_scale_range(scale[0])
            self._check_scale_range(scale[1])
            self._scale = sorted(scale)
        elif isinstance(scale, (int, float)):
            self._check_scale_range(scale)
            scale = abs(scale)
            self._scale = [-scale, scale]
        else:
            raise ValueError(_SCALE_VALIDATION_ERROR + f'Got {scale}')

    @staticmethod
    def _check_scale_range(input_number):
        if input_number > 1.0 or input_number < -1.0:
            raise ValueError(_SCALE_VALIDATION_ERROR + f'Got {input_number}')

    def call(self, image, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        return tf.__internal__.smart_cond.smart_cond(
            training,
            true_fn=lambda: self._brightness_adjust(image),
            false_fn=lambda: image)

    def _brightness_adjust(self, image):
        if self._seed is not None:
            rgb_delta = tf.random.stateless_uniform(
                shape=(3,), seed=[0, self._seed],
                minval=self._scale[0], maxval=self._scale[1])
        else:
            rgb_delta = tf.random.uniform(
                shape=(3,), minval=self._scale[0], maxval=self._scale[1])
        rgb_delta = rgb_delta * 255.0
        input_dtype = image.dtype
        image = tf.cast(image, tf.float32)
        image += rgb_delta
        image = tf.clip_by_value(image, 0.0, 255.0)
        return tf.cast(image, input_dtype)

    def get_config(self):
        config = {
            'scale': self._scale,
            'seed': self._seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
