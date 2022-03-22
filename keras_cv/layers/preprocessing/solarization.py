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

    When created without `threshold` parameter, the layer performs solarization to
    all values. When created with specified `threshold` the layer only augments
    pixels that are above the `threshold` value

    Reference:
    - [AutoAugment: Learning Augmentation Policies from Data](
        https://arxiv.org/abs/1805.09501
    )

    Args:
        threshold: (Optionally) int or float. If specified, only pixel values above this
            threshold will be solarized.

    Usage:

    ```python
        (images, labels), _ = tf.keras.datasets.cifar10.load_data()
        print(images[0, 0, 0])
        # [59 62 63]
        # Note that images are Tensor with values in the range [0, 255]
        solarization = Solarization()
        images = solarization(images)
        print(images[0, 0, 0])
        # [196, 193, 192]
    ```

    Call arguments:
        images: Tensor of type int or float, with pixels in
            range [0, 255] and shape [batch, height, width, channels]
            or [height, width, channels].
    """

    def __init__(self, threshold=None):
        super().__init__()
        self.threshold = threshold

    def call(self, images):
        images = tf.clip_by_value(images, clip_value_min=0, clip_value_max=255)
        if self.threshold is None:
            return self._solarize(images)
        else:
            return self._solarize_above_threshold(images)

    def _solarize(self, images):
        return 255 - images

    def _solarize_above_threshold(self, images):
        return tf.where(images < self.threshold, images, self._solarize(images))

    def get_config(self):
        config = {"threshold": self.threshold}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
