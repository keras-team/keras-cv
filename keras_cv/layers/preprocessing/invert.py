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


class Invert(tf.keras.layers.Layer):
    """Inverts the image pixels.

    Usage:
        ```
        invert = Invert()
        (images, labels), _ = tf.keras.dataset.cifar10.load_data()
        # Note that images are Tensor with values in the range [0, 255]
        images = invert(images)
        ```

    Call arguments:
        images: Tensor of type int or float, pixels in range [0, 255].
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
