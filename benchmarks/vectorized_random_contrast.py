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

from keras_cv.layers import RandomContrast
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.utils import preprocessing as preprocessing_utils


class OldRandomContrast(BaseImageAugmentationLayer):
    """RandomContrast randomly adjusts contrast during training.

    This layer will randomly adjust the contrast of an image or images by a
    random factor. Contrast is adjusted independently for each channel of each
    image during training.

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
      factor: a positive float represented as fraction of value, or a tuple of
        size 2 representing lower and upper bound. When represented as a single
        float, lower = upper. The contrast factor will be randomly picked
        between `[1.0 - lower, 1.0 + upper]`. For any pixel x in the channel,
        the output will be `(x - mean) * factor + mean` where `mean` is the mean
        value of the channel.
      seed: Integer. Used to create a random seed.
    """

    def __init__(self, factor, seed=None, **kwargs):
        super().__init__(seed=seed, force_generator=True, **kwargs)
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
        self.seed = seed

    def get_random_transformation(self, **kwargs):
        return self.factor()

    def augment_image(self, image, transformation, **kwargs):
        contrast_factor = transformation
        output = tf.image.adjust_contrast(
            image, contrast_factor=contrast_factor
        )
        output = tf.clip_by_value(output, 0, 255)
        output.set_shape(image.shape)
        return output

    def augment_label(self, label, transformation, **kwargs):
        return label

    def augment_segmentation_mask(
        self, segmentation_mask, transformation, **kwargs
    ):
        return segmentation_mask

    def augment_bounding_boxes(
        self, bounding_boxes, transformation=None, **kwargs
    ):
        return bounding_boxes

    def get_config(self):
        config = {
            "factor": self.factor_input,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RandomContrastTest(tf.test.TestCase):
    def test_consistency_with_old_impl_rescaled_range(self):
        image_shape = (16, 32, 32, 3)
        fixed_factor = (0.3, -0.3)  # makes lower and upper the same
        image = tf.random.uniform(shape=image_shape)

        layer = RandomContrast(factor=fixed_factor)
        old_layer = OldRandomContrast(factor=fixed_factor)

        output = layer(image)
        old_output = old_layer(image)

        self.assertAllClose(old_output, output)

    def test_consistency_with_old_impl_rgb_range(self):
        image_shape = (16, 32, 32, 3)
        fixed_factor = (0.3, -0.3)  # makes lower and upper the same
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = RandomContrast(factor=fixed_factor)
        old_layer = OldRandomContrast(factor=fixed_factor)

        output = layer(image)
        old_output = old_layer(image)

        self.assertAllClose(old_output, output, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    # Run benchmark
    (x_train, _), _ = keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32)

    num_images = [1000, 2000, 5000, 10000]
    results = {}
    aug_candidates = [RandomContrast, OldRandomContrast]
    aug_args = {"factor": (0.5)}

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

    # Run unit tests
    tf.test.main()
