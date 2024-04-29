# Copyright 2023 The KerasCV Authors
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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    VectorizedBaseImageAugmentationLayer,
)
from keras_cv.src.utils import preprocessing as preprocessing_utils


@keras_cv_export("keras_cv.layers.RandomHue")
class RandomHue(VectorizedBaseImageAugmentationLayer):
    """Randomly adjusts the hue on given images.

    This layer will randomly increase/reduce the hue for the input RGB
    images.

    The image hue is adjusted by converting the image(s) to HSV and rotating the
    hue channel (H) by delta. The image is then converted back to RGB.

    Args:
        factor: A tuple of two floats, a single float or
            `keras_cv.FactorSampler`. `factor` controls the extent to which the
            image hue is impacted. `factor=0.0` makes this layer perform a
            no-op operation, while a value of 1.0 performs the most aggressive
            contrast adjustment available. If a tuple is used, a `factor` is
            sampled between the two values for every image augmented. If a
            single float is used, a value between `0.0` and the passed float is
            sampled. In order to ensure the value is always the same, please
            pass a tuple with two identical floats: `(0.5, 0.5)`.
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written [low, high]. This is
            typically either `[0, 1]` or `[0, 255]` depending on how your
            preprocessing pipeline is set up.
        seed: Integer. Used to create a random seed.

    Example:
    ```python
    (images, labels), _ = keras.datasets.cifar10.load_data()
    random_hue = keras_cv.layers.preprocessing.RandomHue()
    augmented_images = random_hue(images)
    ```
    """

    def __init__(self, factor, value_range, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.factor = preprocessing_utils.parse_factor(
            factor,
        )
        self.value_range = value_range
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        invert = self._random_generator.uniform((batch_size,), 0, 1, tf.float32)
        invert = tf.where(
            invert > 0.5, -tf.ones_like(invert), tf.ones_like(invert)
        )
        # We must scale self.factor() to the range [-0.5, 0.5]. This is because
        # the tf.image operation performs rotation on the hue saturation value
        # orientation. This can be thought of as an angle in the range
        # [-180, 180]
        return invert * self.factor(shape=(batch_size,)) * 0.5

    def augment_ragged_image(self, image, transformation, **kwargs):
        return self.augment_images(
            images=image, transformations=transformation, **kwargs
        )

    def augment_images(self, images, transformations, **kwargs):
        images = preprocessing_utils.transform_value_range(
            images, self.value_range, (0, 1), dtype=self.compute_dtype
        )
        adjust_factors = tf.cast(transformations, images.dtype)
        # broadcast
        adjust_factors = adjust_factors[..., tf.newaxis, tf.newaxis]

        # tf.image.adjust_hue expects floats to be in range [0, 1]
        images = tf.image.rgb_to_hsv(images)
        h_channel = images[..., 0] + adjust_factors
        h_channel = tf.where(h_channel > 1.0, h_channel - 1.0, h_channel)
        h_channel = tf.where(h_channel < 0.0, h_channel + 1.0, h_channel)
        images = tf.stack([h_channel, images[..., 1], images[..., 2]], axis=-1)
        images = tf.image.hsv_to_rgb(images)
        # RandomHue is one of the rare KPLs that needs to clip
        images = tf.clip_by_value(images, 0, 1)
        images = preprocessing_utils.transform_value_range(
            images, (0, 1), self.value_range, dtype=self.compute_dtype
        )
        return images

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        return segmentation_masks

    def augment_bounding_boxes(self, bounding_boxes, transformations, **kwargs):
        return bounding_boxes

    def get_config(self):
        config = {
            "factor": self.factor,
            "value_range": self.value_range,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if isinstance(config["factor"], dict):
            config["factor"] = keras.utils.deserialize_keras_object(
                config["factor"]
            )
        return cls(**config)
