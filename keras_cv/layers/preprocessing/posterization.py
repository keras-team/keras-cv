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

from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.utils.preprocessing import transform_value_range


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class Posterization(BaseImageAugmentationLayer):
    """Reduces the number of bits for each color channel.

    References:
    - [AutoAugment: Learning Augmentation Policies from Data](
        https://arxiv.org/abs/1805.09501
    )
    - [RandAugment: Practical automated data augmentation with a reduced search space](
        https://arxiv.org/abs/1909.13719
    )

    Args:
        value_range: a tuple or a list of two elements. The first value represents
            the lower bound for values in passed images, the second represents the
            upper bound. Images passed to the layer should have values within
            `value_range`. Defaults to `(0, 255)`.
        bits: integer. The number of bits to keep for each channel. Must be a value
            between 1-8.

     Usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    print(images[0, 0, 0])
    # [59 62 63]
    # Note that images are Tensors with values in the range [0, 255] and uint8 dtype
    posterization = Posterization(bits=4, value_range=[0, 255])
    images = posterization(images)
    print(images[0, 0, 0])
    # [48., 48., 48.]
    # NOTE: the layer will output values in tf.float32, regardless of input dtype.
    ```

     Call arguments:
        inputs: input tensor in two possible formats:
            1. single 3D (HWC) image or 4D (NHWC) batch of images.
            2. A dict of tensors where the images are under `"images"` key.
    """

    def __init__(self, value_range, bits, **kwargs):
        super().__init__(**kwargs)

        if not len(value_range) == 2:
            raise ValueError(
                "value_range must be a sequence of two elements. "
                f"Received: {value_range}"
            )

        if not (0 < bits < 9):
            raise ValueError(f"Bits value must be between 1-8. Received bits: {bits}.")

        self._shift = 8 - bits
        self._value_range = value_range

    def augment_image(self, image, transformation=None, **kwargs):
        image = transform_value_range(
            images=image,
            original_range=self._value_range,
            target_range=[0, 255],
        )
        image = tf.cast(image, tf.uint8)

        image = self._posterize(image)

        image = tf.cast(image, self.compute_dtype)
        return transform_value_range(
            images=image,
            original_range=[0, 255],
            target_range=self._value_range,
        )

    def _batch_augment(self, inputs):
        # Skip the use of vectorized_map or map_fn as the implementation is already
        # vectorized
        return self._augment(inputs)

    def _posterize(self, image):
        return tf.bitwise.left_shift(
            tf.bitwise.right_shift(image, self._shift), self._shift
        )

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def get_config(self):
        config = {"bits": 8 - self._shift, "value_range": self._value_range}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
