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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.layers import preprocessing
from keras_cv.src.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    VectorizedBaseImageAugmentationLayer,
)
from keras_cv.src.utils import preprocessing as preprocessing_utils


@keras_cv_export("keras_cv.layers.RandomColorJitter")
class RandomColorJitter(VectorizedBaseImageAugmentationLayer):
    """RandomColorJitter class randomly apply brightness, contrast, saturation
    and hue image processing operation sequentially and randomly on the
    input. It expects input as RGB image. The expected image should be
    `(0-255)` pixel ranges.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `channels_last` format
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `channels_last` format

    Args:
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written [low, high].
            This is typically either `[0, 1]` or `[0, 255]` depending
            on how your preprocessing pipeline is set up.
        brightness_factor: Float or a list/tuple of 2 floats between -1.0
            and 1.0. The factor is used to determine the lower bound and
            upper bound of the brightness adjustment. A float value will be
            chosen randomly between the limits. When -1.0 is chosen, the
            output image will be black, and when 1.0 is chosen, the image
            will be fully white. When only one float is provided, eg, 0.2,
            then -0.2 will be used for lower bound and 0.2 will be used for
            upper bound.
        contrast_factor: A positive float represented as fraction of value,
            or a tuple of size 2 representing lower and upper bound. When
            represented as a single float, lower = upper. The contrast factor
            will be randomly picked between `[1.0 - lower, 1.0 + upper]`.
        saturation_factor: Either a tuple of two floats or a single float.
            `factor` controls the extent to which the image saturation is
            impacted. `factor=0.5` makes this layer perform a no-op operation.
            `factor=0.0` makes the image to be fully grayscale. `factor=1.0`
            makes the image to be fully saturated.
        hue_factor: A tuple of two floats, a single float or
            `keras_cv.FactorSampler`. `factor` controls the extent to which the
            image sharpness is impacted. `factor=0.0` makes this layer perform
            a no-op operation, while a value of 1.0 performs the most aggressive
            contrast adjustment available. If a tuple is used, a `factor` is
            sampled between the two values for every image augmented. If a
            single float is used, a value between `0.0` and the passed float is
            sampled. In order to ensure the value is always the same, please
            pass a tuple with two identical floats: `(0.5, 0.5)`.
        seed: Integer. Used to create a random seed.

    Example:
    ```python
    (images, labels), _ = keras.datasets.cifar10.load_data()
    color_jitter = keras_cv.layers.RandomColorJitter(
            value_range=(0, 255),
            brightness_factor=(-0.2, 0.5),
            contrast_factor=(0.5, 0.9),
            saturation_factor=(0.5, 0.9),
            hue_factor=(0.5, 0.9),
    )
    augmented_images = color_jitter(images)
    ```
    """

    def __init__(
        self,
        value_range,
        brightness_factor,
        contrast_factor,
        saturation_factor,
        hue_factor,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.value_range = value_range
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.saturation_factor = saturation_factor
        self.hue_factor = hue_factor
        self.seed = seed

        self.random_brightness = preprocessing.RandomBrightness(
            factor=self.brightness_factor, value_range=(0, 255), seed=self.seed
        )
        self.random_contrast = preprocessing.RandomContrast(
            factor=self.contrast_factor, value_range=(0, 255), seed=self.seed
        )
        self.random_saturation = preprocessing.RandomSaturation(
            factor=self.saturation_factor, seed=self.seed
        )
        self.random_hue = preprocessing.RandomHue(
            factor=self.hue_factor, value_range=(0, 255), seed=self.seed
        )

    def augment_ragged_image(self, image, transformation, **kwargs):
        return self.augment_images(
            images=image, transformations=transformation, **kwargs
        )

    def augment_images(self, images, transformations=None, **kwargs):
        images = preprocessing_utils.transform_value_range(
            images,
            original_range=self.value_range,
            target_range=(0, 255),
            dtype=self.compute_dtype,
        )
        images = self.random_brightness(images)
        images = self.random_contrast(images)
        images = self.random_saturation(images)
        images = self.random_hue(images)
        images = preprocessing_utils.transform_value_range(
            images,
            original_range=(0, 255),
            target_range=self.value_range,
            dtype=self.compute_dtype,
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
        config = super().get_config()
        config.update(
            {
                "value_range": self.value_range,
                "brightness_factor": self.brightness_factor,
                "contrast_factor": self.contrast_factor,
                "saturation_factor": self.saturation_factor,
                "hue_factor": self.hue_factor,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
