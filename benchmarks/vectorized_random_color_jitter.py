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

import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras_cv.layers import preprocessing
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.utils import preprocessing as preprocessing_utils


class OldRandomColorJitter(BaseImageAugmentationLayer):
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
        value_range:  the range of values the incoming images will have.
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
            contrast adjustment available.  If a tuple is used, a `factor` is
            sampled between the two values for every image augmented. If a
            single float is used, a value between `0.0` and the passed float is
            sampled. In order to ensure the value is always the same, please
            pass a tuple with two identical floats: `(0.5, 0.5)`.
        seed: Integer. Used to create a random seed.

    Usage:
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
            factor=self.contrast_factor, seed=self.seed
        )
        self.random_saturation = preprocessing.RandomSaturation(
            factor=self.saturation_factor, seed=self.seed
        )
        self.random_hue = preprocessing.RandomHue(
            factor=self.hue_factor, value_range=(0, 255), seed=self.seed
        )

    def augment_image(self, image, transformation=None, **kwargs):
        image = preprocessing_utils.transform_value_range(
            image,
            original_range=self.value_range,
            target_range=(0, 255),
            dtype=self.compute_dtype,
        )
        image = self.random_brightness(image)
        image = self.random_contrast(image)
        image = self.random_saturation(image)
        image = self.random_hue(image)
        image = preprocessing_utils.transform_value_range(
            image,
            original_range=(0, 255),
            target_range=self.value_range,
            dtype=self.compute_dtype,
        )
        return image

    def augment_bounding_boxes(self, bounding_boxes, **kwargs):
        return bounding_boxes

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def augment_segmentation_mask(
        self, segmentation_mask, transformation, **kwargs
    ):
        return segmentation_mask

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


if __name__ == "__main__":
    # Run benchmark
    (x_train, _), _ = keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32)

    num_images = [1000, 2000, 5000, 10000]
    results = {}
    aug_candidates = [preprocessing.RandomColorJitter, OldRandomColorJitter]
    aug_args = {
        "value_range": (0, 255),
        "brightness_factor": (-0.2, 0.5),
        "contrast_factor": (0.5, 0.9),
        "saturation_factor": (0.5, 0.9),
        "hue_factor": (0.5, 0.9),
    }

    for aug in aug_candidates:
        # Eager Mode
        c = aug.__name__
        layer = aug(**aug_args)
        runtimes = []
        print(f"Timing {c}")

        for n_images in num_images:
            # warmup
            layer(x_train[:n_images])

            t0 = time.time()
            r1 = layer(x_train[:n_images])
            t1 = time.time()
            runtimes.append(t1 - t0)
            print(f"Runtime for {c}, n_images={n_images}: {t1-t0}")
        results[c] = runtimes

        # Graph Mode
        c = aug.__name__ + " Graph Mode"
        layer = aug(**aug_args)

        @tf.function()
        def apply_aug(inputs):
            return layer(inputs)

        runtimes = []
        print(f"Timing {c}")

        for n_images in num_images:
            # warmup
            apply_aug(x_train[:n_images])

            t0 = time.time()
            r1 = apply_aug(x_train[:n_images])
            t1 = time.time()
            runtimes.append(t1 - t0)
            print(f"Runtime for {c}, n_images={n_images}: {t1-t0}")
        results[c] = runtimes

        # XLA Mode
        c = aug.__name__ + " XLA Mode"
        layer = aug(**aug_args)

        @tf.function(jit_compile=True)
        def apply_aug(inputs):
            return layer(inputs)

        runtimes = []
        print(f"Timing {c}")

        for n_images in num_images:
            # warmup
            apply_aug(x_train[:n_images])

            t0 = time.time()
            r1 = apply_aug(x_train[:n_images])
            t1 = time.time()
            runtimes.append(t1 - t0)
            print(f"Runtime for {c}, n_images={n_images}: {t1-t0}")
        results[c] = runtimes

    plt.figure()
    for key in results:
        plt.plot(num_images, results[key], label=key)
        plt.xlabel("Number images")

    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.savefig("comparison.png")

    # So we can actually see more relevant margins
    del results[aug_candidates[1].__name__]
    plt.figure()
    for key in results:
        plt.plot(num_images, results[key], label=key)
        plt.xlabel("Number images")

    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.savefig("comparison_no_old_eager.png")
