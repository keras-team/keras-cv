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
from keras_cv.src.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    VectorizedBaseImageAugmentationLayer,
)
from keras_cv.src.utils import preprocessing as preprocessing_utils


@keras_cv_export("keras_cv.layers.RandomContrast")
class RandomContrast(VectorizedBaseImageAugmentationLayer):
    """RandomContrast randomly adjusts contrast.

    This layer will randomly adjust the contrast of an image or images by a
    random factor. Contrast is adjusted independently for each channel of each
    image.

    For each channel, this layer computes the mean of the image pixels in the
    channel and then adjusts each component `x` of each pixel to
    `(x - mean) * contrast_factor + mean`.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    in integer or floating point dtype. By default, the layer will output
    floats. The output value will be clipped to the range `[0, 255]`, the valid
    range of RGB colors.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.

    Args:
        value_range: A tuple or a list of two elements. The first value
            represents the lower bound for values in passed images, the second
            represents the upper bound. Images passed to the layer should have
            values within `value_range`.
        factor: A positive float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound. When represented as a
            single float, lower = upper. The contrast factor will be randomly
            picked between `[1.0 - lower, 1.0 + upper]`. For any pixel x in the
            channel, the output will be `(x - mean) * factor + mean` where
            `mean` is the mean value of the channel.
        seed: Integer. Used to create a random seed.

    Example:
    ```python
    (images, labels), _ = keras.datasets.cifar10.load_data()
    random_contrast = keras_cv.layers.preprocessing.RandomContrast()
    augmented_images = random_contrast(images)
    ```
    """

    def __init__(self, value_range, factor, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        if isinstance(factor, (tuple, list)):
            min = 1 - factor[0]
            max = 1 + factor[1]
        else:
            min = 1 - factor
            max = 1 + factor
        self.factor_input = factor
        self.factor = preprocessing_utils.parse_factor(
            (min, max), min_value=-1, max_value=2
        )
        self.value_range = value_range
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        return self.factor(shape=(batch_size, 1, 1, 1))

    def augment_ragged_image(self, image, transformation, **kwargs):
        return self.augment_images(
            images=image, transformations=transformation, **kwargs
        )

    def augment_images(self, images, transformations, **kwargs):
        contrast_factors = tf.cast(transformations, dtype=images.dtype)
        means = tf.reduce_mean(images, axis=(1, 2), keepdims=True)

        images = (images - means) * contrast_factors + means
        images = tf.clip_by_value(
            images, self.value_range[0], self.value_range[1]
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
            "factor": self.factor_input,
            "value_range": self.value_range,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
