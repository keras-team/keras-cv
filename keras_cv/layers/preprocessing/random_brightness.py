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

from keras_cv.layers.preprocessing.vectorized_base_image_augmentation_layer import (
    VectorizedBaseImageAugmentationLayer,
)
from keras_cv.utils import preprocessing as preprocessing_utils


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RandomBrightness(VectorizedBaseImageAugmentationLayer):
    """A preprocessing layer which randomly adjusts brightness during training.
    This layer will randomly increase/reduce the brightness for the input RGB
    images.

    At inference time, the output will be identical to the input.
    Call the layer with `training=True` to adjust the brightness of the input.

    Note that different brightness adjustment factors
    will be apply to each the images in the batch.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `channels_last` format
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `channels_last` format

    Args:
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written (low, high).
            This is typically either `(0, 1)` or `(0, 255)` depending
            on how your preprocessing pipeline is setup. The brightness
            adjustment will be scaled to this range, and the output values will
            be clipped to this range.
        factor: Float or a list/tuple of 2 floats between -1.0 and 1.0. The
            factor is used to determine the lower bound and upper bound of the
            brightness adjustment. A float value will be chosen randomly between
            the limits. When -1.0 is chosen, the output image will be black, and
            when 1.0 is chosen, the image will be fully white. When only one float
            is provided, eg, 0.2, then -0.2 will be used for lower bound and 0.2
            will be used for upper bound.
       seed: optional integer, for fixed RNG behavior.

    Usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    random_brightness = keras_cv.layers.preprocessing.RandomBrightness(
        value_range=(0, 255), factor=0.5
    )
    augmented_images = random_brightness(images)
    ```
    """

    def __init__(self, value_range, factor, seed=None, **kwargs):
        super().__init__(seed=seed, force_generator=True, **kwargs)
        if isinstance(factor, float) or isinstance(factor, int):
            factor = (-factor, factor)
        self.factor = preprocessing_utils.parse_factor(
            factor, min_value=-1, max_value=1
        )
        self.value_range = value_range
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        rgb_delta_shape = (batch_size, 1, 1, 1)
        random_rgb_deltas = self.factor(shape=rgb_delta_shape)
        random_rgb_deltas = random_rgb_deltas * (
            self.value_range[1] - self.value_range[0]
        )
        return random_rgb_deltas

    def augment_ragged_image(self, image, transformation, **kwargs):
        return self.augment_images(
            images=image, transformations=transformation, **kwargs
        )

    def augment_images(self, images, transformations, **kwargs):
        rgb_deltas = tf.cast(transformations, images.dtype)
        images += rgb_deltas
        return tf.clip_by_value(
            images, self.value_range[0], self.value_range[1]
        )

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
            "value_range": self.value_range,
            "factor": self.factor,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if isinstance(config["factor"], dict):
            config["factor"] = tf.keras.utils.deserialize_keras_object(
                config["factor"]
            )
        return cls(**config)
