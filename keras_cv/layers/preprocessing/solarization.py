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

from keras_cv.utils import preprocessing


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class Solarization(tf.keras.__internal__.layers.BaseImageAugmentationLayer):
    """Applies (max_value - pixel + min_value) for each pixel in the image.

    When created without `threshold` parameter, the layer performs solarization to
    all values. When created with specified `threshold` the layer only augments
    pixels that are above the `threshold` value

    Reference:
    - [AutoAugment: Learning Augmentation Policies from Data](
        https://arxiv.org/abs/1805.09501
    )
    - [RandAugment](https://arxiv.org/pdf/1909.13719.pdf)

    Args:
        addition: (Optional) int or float.  If specified, this value is added to each
            pixel before solarization and thresholding.  The addition value should be
            scaled accoring to the value range (0, 255).  Defaults to 0.0.
        threshold: (Optional) int or float. If specified, only pixel values above this
            threshold will be solarized.
        value_range: a tuple or a list of two elements. The first value represents
            the lower bound for values in passed images, the second represents the
            upper bound. Images passed to the layer should have values within
            `value_range`. Defaults to `(0, 255)`.

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

    def __init__(self, addition=0.0, threshold=0.0, value_range=(0, 255), **kwargs):
        super().__init__(**kwargs)
        self.addition = addition
        self.threshold = threshold
        self.value_range = value_range

    def augment_image(self, image, transformation=None):
        image = preprocessing.transform_value_range(
            image, original_range=self.value_range, target_range=(0, 255)
        )
        result = image + self.addition
        result = tf.clip_by_value(result, 0, 255)
        result = self._solarize(result)
        result = preprocessing.transform_value_range(
            result, original_range=(0, 255), target_range=self.value_range
        )
        return result

    def _solarize(self, images):
        return tf.where(images < self.threshold, images, 255 - images)

    def get_config(self):
        config = {
            "threshold": self.threshold,
            "addition": self.addition,
            "value_range": self.value_range,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
