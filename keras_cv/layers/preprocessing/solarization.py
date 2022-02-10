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

    def __init__(self, min_value=0, max_value=255):
        super().__init__()

        assert min_value < max_value, (
            "`min_value` should be smaller than `max_value`. "
            f"Received: min_value: {min_value}, max_value: {max_value}"
        )

        self.min_value = min_value
        self.max_value = max_value

    def call(self, images):
        return self.max_value - images + self.min_value

    def get_config(self):
        config = {
            "min_value": self.min_value,
            "max_value": self.max_value,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
