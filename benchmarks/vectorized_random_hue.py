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

from keras_cv.layers import RandomHue
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.utils import preprocessing as preprocessing_utils


class OldRandomHue(BaseImageAugmentationLayer):
    """Randomly adjusts the hue on given images.

    This layer will randomly increase/reduce the hue for the input RGB
    images. At inference time, the output will be identical to the input.
    Call the layer with `training=True` to adjust the brightness of the input.

    The image hue is adjusted by converting the image(s) to HSV and rotating the
    hue channel (H) by delta. The image is then converted back to RGB.

    Args:
        factor: A tuple of two floats, a single float or `keras_cv.FactorSampler`.
            `factor` controls the extent to which the image hue is impacted.
            `factor=0.0` makes this layer perform a no-op operation, while a value of
            1.0 performs the most aggressive contrast adjustment available.  If a tuple
            is used, a `factor` is sampled between the two values for every image
            augmented.  If a single float is used, a value between `0.0` and the passed
            float is sampled.  In order to ensure the value is always the same, please
            pass a tuple with two identical floats: `(0.5, 0.5)`.
        value_range:  the range of values the incoming images will have.
            Represented as a two number tuple written [low, high].
            This is typically either `[0, 1]` or `[0, 255]` depending
            on how your preprocessing pipeline is setup.
        seed: Integer. Used to create a random seed.

    """

    def __init__(self, factor, value_range, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.factor = preprocessing_utils.parse_factor(
            factor,
        )
        self.value_range = value_range
        self.seed = seed

    def get_random_transformation(self, **kwargs):
        invert = preprocessing_utils.random_inversion(self._random_generator)
        # We must scale self.factor() to the range [-0.5, 0.5].  This is because the
        # tf.image operation performs rotation on the hue saturation value orientation.
        # This can be thought of as an angle in the range [-180, 180]
        return invert * self.factor() * 0.5

    def augment_image(self, image, transformation=None, **kwargs):
        image = preprocessing_utils.transform_value_range(
            image, self.value_range, (0, 1), dtype=self.compute_dtype
        )

        # tf.image.adjust_hue expects floats to be in range [0, 1]
        image = tf.image.adjust_hue(image, delta=transformation)
        # RandomHue is one of the rare KPLs that needs to clip
        image = tf.clip_by_value(image, 0, 1)
        image = preprocessing_utils.transform_value_range(
            image, (0, 1), self.value_range, dtype=self.compute_dtype
        )
        return image

    def augment_bounding_boxes(self, bounding_boxes, **kwargs):
        return bounding_boxes

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def augment_segmentation_mask(self, segmentation_mask, transformation, **kwargs):
        return segmentation_mask

    def get_config(self):
        config = {
            "factor": self.factor,
            "value_range": self.value_range,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RandomHueTest(tf.test.TestCase):
    def test_consistency_with_old_impl_rescaled_range(self):
        image_shape = (16, 32, 32, 3)
        fixed_factor = (0.8, 0.8)
        fixed_seed = 2023
        image = tf.random.uniform(shape=image_shape)

        layer = RandomHue(fixed_factor, (0, 1), fixed_seed)
        old_layer = OldRandomHue(fixed_factor, (0, 1), fixed_seed)

        output = layer(image)
        old_output = old_layer(image)

        self.assertAllClose(old_output, output)

    def test_consistency_with_old_impl_rgb_range(self):
        image_shape = (16, 32, 32, 3)
        fixed_factor = (0.8, 0.8)
        fixed_seed = 2023
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = RandomHue(fixed_factor, (0, 255), fixed_seed)
        old_layer = OldRandomHue(fixed_factor, (0, 255), fixed_seed)

        output = layer(image)
        old_output = old_layer(image)

        self.assertAllClose(old_output, output, atol=1e-3, rtol=1e-5)


if __name__ == "__main__":
    # Run benchmark
    (x_train, _), _ = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32)

    num_images = [1000, 2000, 5000, 10000]
    results = {}
    aug_candidates = [RandomHue, OldRandomHue]
    aug_args = {"factor": (0.5), "value_range": (0, 255)}

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
        # OldRandomHue fails to run jit_compile=True

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
