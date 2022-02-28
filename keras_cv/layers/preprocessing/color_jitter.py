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


class ColorJitter(layers.Layer):
    """ColorJitter class randomly apply brightness, contrast, saturation 
    and hue image processing operation sequentially and randomly. 
    It expects input RGB image. The expected images should be `(0-255)` pixel ranges.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `channels_last` format
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `channels_last` format

    Args:
        brightness_factor: A non-negative single float value. It will be used 
        as `[-brightness_factor, brightness_factor)` to randomly pick a value 
        from uniform distribution. Default: 0.5.

        contrast_factor: A non-negative scalar or tuple or list of two upper 
        and lower bound number. If factor is a single value, the range will 
        be `(0, contrast_factor)`; otherwise (contrast_factor[0], contrast_factor[1])
        as lower and upper bound respectively. It will be used to randomly pick a value 
        from uniform distribution. Default: (0.5, 0.9).

        saturation_factor: A non-negative scalar or tuple or list of two upper 
        and lower bound number. If factor is a single value, the range will 
        be `(0, saturation_factor)`; otherwise (saturation_factor[0], saturation_factor[1])
        as lower and upper bound respectively. Default: (0.5, 0.9). It will be used 
        to randomly pick a value from uniform distribution. Default: (0.5, 0.9).

        hue_factor: A non-negative single float value. It will be used 
        as `[-hue_factor, hue_factor)` to randomly pick a value 
        from uniform distribution. Default: 0.5.
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
    color_jitter = keras_cv.layers.ColorJitter()
    augmented_images = color_jitter(images)
    ```
    """

    def __init__(
        self,
        brightness_factor=0.5,
        contrast_factor=(0.5, 0.9),
        saturation_factor=(0.5, 0.9),
        hue_factor=0.5,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seed = seed
        self.brightness_factor = self._check_factor_limit(
            brightness_factor, name="brightness"
        )
        self.contrast_factor = self._check_factor_limit(
            contrast_factor, name="contrast"
        )
        self.saturation_factor = self._check_factor_limit(
            saturation_factor, name="saturation"
        )
        self.hue_factor = self._check_factor_limit(hue_factor, name="hue")

    def _check_factor_limit(self, factor, name):
        if isinstance(factor, (int, float)):
            if factor < 0:
                raise TypeError(
                    "The factor value should be non-negative scalar or tuple "
                    f"or list of two upper and lower bound number. Received: {factor}"
                )
            if name == "brightness" or name == "hue":
                return abs(factor)
            return (0, abs(factor))
        elif isinstance(factor, (tuple, list)) and len(factor) == 2:
            if name == "brightness" or name == "hue":
                raise ValueError(
                    "The factor limit for brightness and hue, it should be a single "
                    f"non-negative scaler. Received: {factor} for {name}"
                )
            return sorted(factor)
        else:
            raise TypeError(
                "The factor value should be non-negative scalar or tuple "
                f"or list of two upper and lower bound number. Received: {factor}"
            )

    def _color_jitter(self, images):
        original_dtype = images.dtype
        images = tf.cast(images, dtype=tf.float32)

        brightness = tf.image.random_brightness(
            images, max_delta=self.brightness_factor * 255.0, seed=self.seed
        )
        brightness = tf.clip_by_value(brightness, 0.0, 255.0)

        contrast = tf.image.random_contrast(
            brightness,
            lower=self.contrast_factor[0],
            upper=self.contrast_factor[1],
            seed=self.seed,
        )
        saturation = tf.image.random_saturation(
            contrast,
            lower=self.saturation_factor[0],
            upper=self.saturation_factor[1],
            seed=self.seed,
        )
        hue = tf.image.random_hue(saturation, max_delta=self.hue_factor, seed=self.seed)
        return tf.cast(hue, original_dtype)

    def call(self, images, training=True):
        if training:
            return self._color_jitter(images)
        else:
            return images

    def get_config(self):
        config = {
            "brightness_factor": self.brightness_factor,
            "contrast_factor": self.contrast_factor,
            "saturation_factor": self.saturation_factor,
            "hue_factor": self.hue_factor,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
