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


@keras_cv_export("keras_cv.layers.RandomSaturation")
class RandomSaturation(VectorizedBaseImageAugmentationLayer):
    """Randomly adjusts the saturation on given images.

    This layer will randomly increase/reduce the saturation for the input RGB
    images.

    Args:
        factor: A tuple of two floats, a single float or
            `keras_cv.FactorSampler`. `factor` controls the extent to which the
            image saturation is impacted. `factor=0.5` makes this layer perform
            a no-op operation. `factor=0.0` makes the image to be fully
            grayscale. `factor=1.0` makes the image to be fully saturated.
            Values should be between `0.0` and `1.0`. If a tuple is used, a
            `factor` is sampled between the two values for every image
            augmented. If a single float is used, a value between `0.0` and the
            passed float is sampled. In order to ensure the value is always the
            same, please pass a tuple with two identical floats: `(0.5, 0.5)`.
        seed: Integer. Used to create a random seed.

    Example:
    ```python
    (images, labels), _ = keras.datasets.cifar10.load_data()
    random_saturation = keras_cv.layers.preprocessing.RandomSaturation()
    augmented_images = random_saturation(images)
    ```
    """

    def __init__(self, factor, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.factor = preprocessing_utils.parse_factor(
            factor,
            min_value=0.0,
            max_value=1.0,
        )
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        return self.factor(shape=(batch_size,))

    def augment_ragged_image(self, image, transformation, **kwargs):
        return self.augment_images(
            images=image, transformations=transformation, **kwargs
        )

    def augment_images(self, images, transformations, **kwargs):
        # Convert the factor range from [0, 1] to [0, +inf]. Note that the
        # tf.image.adjust_saturation is trying to apply the following math
        # formula `output_saturation = input_saturation * factor`. We use the
        # following method to the do the mapping.
        # `y = x / (1 - x)`.
        # This will ensure:
        #   y = +inf when x = 1 (full saturation)
        #   y = 1 when x = 0.5 (no augmentation)
        #   y = 0 when x = 0 (full gray scale)

        # Convert the transformation to tensor in case it is a float. When
        # transformation is 1.0, then it will result in to divide by zero error,
        # but it will be handled correctly when it is a one tensor.
        transformations = tf.convert_to_tensor(transformations)
        adjust_factors = transformations / (1 - transformations)
        adjust_factors = tf.cast(adjust_factors, dtype=images.dtype)

        images = tf.image.rgb_to_hsv(images)
        s_channel = tf.multiply(
            images[..., 1], adjust_factors[..., tf.newaxis, tf.newaxis]
        )
        s_channel = tf.clip_by_value(
            s_channel, clip_value_min=0.0, clip_value_max=1.0
        )
        images = tf.stack([images[..., 0], s_channel, images[..., 2]], axis=-1)
        images = tf.image.hsv_to_rgb(images)
        return images

    def augment_bounding_boxes(
        self, bounding_boxes, transformation=None, **kwargs
    ):
        return bounding_boxes

    def augment_labels(self, labels, transformations=None, **kwargs):
        return labels

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        return segmentation_masks

    def get_config(self):
        config = {
            "factor": self.factor,
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
