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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    VectorizedBaseImageAugmentationLayer,
)
from keras_cv.src.utils import preprocessing


@keras_cv_export("keras_cv.layers.AutoContrast")
class AutoContrast(VectorizedBaseImageAugmentationLayer):
    """Performs the AutoContrast operation on an image.

    Auto contrast stretches the values of an image across the entire available
    `value_range`. This makes differences between pixels more obvious. An
    example of this is if an image only has values `[0, 1]` out of the range
    `[0, 255]`, auto contrast will change the `1` values to be `255`.

    Args:
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written [low, high].
            This is typically either `[0, 1]` or `[0, 255]` depending
            on how your preprocessing pipeline is set up.
    """

    def __init__(
        self,
        value_range,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.value_range = value_range

    def augment_images(self, images, transformations=None, **kwargs):
        original_images = images
        images = preprocessing.transform_value_range(
            images,
            original_range=self.value_range,
            target_range=(0, 255),
            dtype=self.compute_dtype,
        )

        low = tf.reduce_min(images, axis=(1, 2), keepdims=True)
        high = tf.reduce_max(images, axis=(1, 2), keepdims=True)
        scale = 255.0 / (high - low)
        offset = -low * scale

        images = images * scale + offset
        result = tf.clip_by_value(images, 0.0, 255.0)
        result = preprocessing.transform_value_range(
            result,
            original_range=(0, 255),
            target_range=self.value_range,
            dtype=self.compute_dtype,
        )
        # don't process NaN channels
        result = tf.where(tf.math.is_nan(result), original_images, result)
        return result

    def augment_bounding_boxes(self, bounding_boxes, **kwargs):
        return bounding_boxes

    def augment_labels(self, labels, transformations=None, **kwargs):
        return labels

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        return segmentation_masks

    def augment_keypoints(self, keypoints, transformations, **kwargs):
        return keypoints

    def augment_targets(self, targets, transformations, **kwargs):
        return targets

    def augment_ragged_image(self, image, transformation, **kwargs):
        return self.augment_images(
            image, transformations=transformation, **kwargs
        )

    def get_config(self):
        config = super().get_config()
        config.update({"value_range": self.value_range})
        return config
