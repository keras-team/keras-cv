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


class Posterization(tf.keras.layers.Layer):
    """Reduces the number of bits for each color channel.

    References:
    - [AutoAugment: Learning Augmentation Policies from Data](
        https://arxiv.org/abs/1805.09501
    )
    - [RandAugment: Practical automated data augmentation with a reduced search space](
        https://arxiv.org/abs/1909.13719
    )

    Args:
        bits: integer. The number of bits to keep for each channel. Must be a value
            between 1-8.

     Usage:

    ```python
        (images, labels), _ = tf.keras.datasets.cifar10.load_data()
        print(images[0, 0, 0])
        # [59 62 63]
        # Note that images are Tensors with values in the range [0, 255] and uint8 dtype
        posterization = Posterization(bits=4)
        images = posterization(images)
        print(images[0, 0, 0])
        # [48, 48, 48]
    ```

     Call arguments:
        images: Tensor with pixels in range [0, 255] and shape
            [batch, height, width, channels] or [height, width, channels].
    """

    def __init__(self, bits: int):
        super().__init__()
        assert 0 < bits < 9, f"Bits value must be between 1-8. Received bits: {bits}."
        self.shift = 8 - bits

    def call(self, images):
        dtype = images.dtype
        images = tf.cast(images, tf.uint8)
        images = tf.bitwise.left_shift(
            tf.bitwise.right_shift(images, self.shift), self.shift
        )
        return tf.cast(images, dtype)

    def get_config(self):
        config = {"shift": self.shift}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
