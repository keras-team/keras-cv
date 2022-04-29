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
        value_range: a tuple or a list of two elements. The first value represents
            the lower bound for values in passed images, the second represents the
            upper bound. Images passed to the layer should have values within
            `value_range`.
        addition_factor: (Optional)  A tuple of two floats, a single float or a
            `keras_cv.FactorSampler`. For each augmented image a value is sampled
            from the provided range. If a float is passed, the range is interpreted as
            `(0, addition_factor)`. If specified, this value is added to each pixel
            before solarization and thresholding.  The addition value should be scaled
            according to the value range (0, 255). Defaults to 0.0.
        threshold_factor: (Optional)  A tuple of two floats, a single float or a
            `keras_cv.FactorSampler`. For each augmented image a value is sampled
            from the provided range. If a float is passed, the range is interpreted as
            `(0, threshold_factor)`. If specified, only pixel values above this
            threshold will be solarized.
        seed: Integer. Used to create a random seed.

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

    def __init__(
        self,
        value_range,
        addition_factor=0.0,
        threshold_factor=0.0,
        seed=None,
        **kwargs
    ):
        super().__init__(seed=seed, **kwargs)
        self.addition_factor = preprocessing.parse_factor(
            addition_factor, max_value=255, seed=seed, param_name="addition_factor"
        )
        self.threshold_factor = preprocessing.parse_factor(
            threshold_factor, max_value=255, seed=seed, param_name="threshold_factor"
        )
        self.value_range = value_range

    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        return (self.addition_factor(), self.threshold_factor())

    def augment_image(self, image, transformation=None):
        (addition, threshold) = transformation
        image = preprocessing.transform_value_range(
            image, original_range=self.value_range, target_range=(0, 255)
        )
        result = image + addition
        result = tf.clip_by_value(result, 0, 255)
        result = tf.where(result < threshold, result, 255 - result)
        result = preprocessing.transform_value_range(
            result, original_range=(0, 255), target_range=self.value_range
        )
        return result

    def augment_label(self, label, transformation=None):
        return label

    def get_config(self):
        config = {
            "threshold_factor": self.threshold_factor,
            "addition_factor": self.addition_factor,
            "value_range": self.value_range,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
