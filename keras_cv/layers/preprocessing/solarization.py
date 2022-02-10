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


class Solarization(tf.keras.layers.Layer):
    """Applies (max_value - pixel + min_value) for each pixel in the image.

    Args:
        min_value: int or float. Lower bound of input pixel values.
        max_value: int or float. Upper bound of input pixel values.
        threshold: (Optionally) int or float. If specified, only pixel values above this
            threshold will be solarized.

    Usage:

    ```python
        (images, labels), _ = tf.keras.datasets.cifar10.load_data()
        print(images[0, 0, 0])  # [59 62 63]
        # Note that images are Tensor with values in the range [0, 255]
        solarization = Solarization(min_value=0, max_value=255)
        images = solarization(images)
        print(images[0, 0, 0])  # [196, 193, 192]
    ```

    Call arguments:
        images: Tensor of type int or float, with pixels in
            range [`min_value`, `max_value`] and shape [batch, height, width, channels]
            or [height, width, channels].
    """

    def __init__(self, min_value, max_value, threshold=None):
        super().__init__()

        assert min_value < max_value, (
            "`min_value` should be smaller than `max_value`. "
            f"Received: min_value: {min_value}, max_value: {max_value}"
        )

        self.min_value = min_value
        self.max_value = max_value
        self.threshold = threshold

    def call(self, images):
        images = tf.clip_by_value(
            images, clip_value_min=self.min_value, clip_value_max=self.max_value
        )
        if self.threshold is None:
            return self._solarize(images)
        else:
            return self._solarize_above_threshold(images)

    def _solarize(self, images):
        return self.max_value - images + self.min_value

    def _solarize_above_threshold(self, images):
        return tf.where(images < self.threshold, images, self._solarize(images))

    def get_config(self):
        config = {
            "min_value": self.min_value,
            "max_value": self.max_value,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
