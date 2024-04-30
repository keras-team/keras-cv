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
from keras_cv.src.utils import preprocessing


@keras_cv_export("keras_cv.layers.RandomSharpness")
class RandomSharpness(VectorizedBaseImageAugmentationLayer):
    """Randomly performs the sharpness operation on given images.

    The sharpness operation first performs a blur operation, then blends between
    the original image and the blurred image. This operation makes the edges of
    an image less sharp than they were in the original image.

    References:
        - [PIL](https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html)

    Args:
        factor: A tuple of two floats, a single float or
            `keras_cv.FactorSampler`. `factor` controls the extent to which the
            image sharpness is impacted. `factor=0.0` makes this layer perform a
            no-op operation, while a value of 1.0 uses the sharpened result
            entirely. Values between 0 and 1 result in linear interpolation
            between the original image and the sharpened image. Values should be
            between `0.0` and `1.0`. If a tuple is used, a `factor` is sampled
            between the two values for every image augmented. If a single float
            is used, a value between `0.0` and the passed float is sampled. In
            order to ensure the value is always the same, please pass a tuple
            with two identical floats: `(0.5, 0.5)`.
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written [low, high].
            This is typically either `[0, 1]` or `[0, 255]` depending
            on how your preprocessing pipeline is set up.
    """  # noqa: E501

    def __init__(
        self,
        factor,
        value_range,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.value_range = value_range
        self.factor = preprocessing.parse_factor(factor)
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        return self.factor(
            shape=(batch_size, 1, 1, 1), dtype=self.compute_dtype
        )

    def augment_images(self, images, transformations, **kwargs):
        images = preprocessing.transform_value_range(
            images,
            original_range=self.value_range,
            target_range=(0, 255),
            dtype=self.compute_dtype,
        )
        original_images = images

        # [1 1 1]
        # [1 5 1]
        # [1 1 1]
        # all divided by 13 is the default 3x3 gaussian smoothing kernel.
        # Correlating or Convolving with this filter is equivalent to performing
        # a gaussian blur.
        kernel = (
            tf.constant(
                [[1, 1, 1], [1, 5, 1], [1, 1, 1]],
                dtype=self.compute_dtype,
                shape=[3, 3, 1, 1],
            )
            / 13.0
        )

        # Tile across channel dimension.
        channels = tf.shape(images)[-1]
        kernel = tf.tile(kernel, [1, 1, channels, 1])
        strides = [1, 1, 1, 1]

        smoothed_image = tf.nn.depthwise_conv2d(
            images, kernel, strides, padding="VALID", dilations=[1, 1]
        )
        smoothed_image = tf.clip_by_value(smoothed_image, 0.0, 255.0)

        # For the borders of the resulting image, fill in the values of the
        # original image.
        mask = tf.ones_like(smoothed_image)
        padded_mask = tf.pad(mask, [[0, 0], [1, 1], [1, 1], [0, 0]])
        padded_smoothed_image = tf.pad(
            smoothed_image, [[0, 0], [1, 1], [1, 1], [0, 0]]
        )

        result = tf.where(
            tf.equal(padded_mask, 1), padded_smoothed_image, original_images
        )
        # Blend the final result.
        result = preprocessing.blend(original_images, result, transformations)
        result = preprocessing.transform_value_range(
            result,
            original_range=(0, 255),
            target_range=self.value_range,
            dtype=self.compute_dtype,
        )
        return result

    def augment_bounding_boxes(self, bounding_boxes, transformations, **kwargs):
        return bounding_boxes

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        return segmentation_masks

    def augment_keypoints(self, keypoints, transformations, **kwargs):
        return keypoints

    def augment_ragged_image(self, image, transformation, **kwargs):
        images = tf.expand_dims(image, axis=0)
        new_transformation = tf.expand_dims(transformation, axis=0)
        output = self.augment_images(images, new_transformation)
        return tf.squeeze(output, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "factor": self.factor,
                "value_range": self.value_range,
                "seed": self.seed,
            }
        )
        return config
