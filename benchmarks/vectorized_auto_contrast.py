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
import tensorflow as tf
import tensorflow.keras as keras

from keras_cv.layers import AutoContrast
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.utils import preprocessing


class OldAutoContrast(BaseImageAugmentationLayer):
    """Performs the AutoContrast operation on an image.

    Auto contrast stretches the values of an image across the entire available
    `value_range`.  This makes differences between pixels more obvious.  An example of
    this is if an image only has values `[0, 1]` out of the range `[0, 255]`, auto
    contrast will change the `1` values to be `255`.

    Args:
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written [low, high].
            This is typically either `[0, 1]` or `[0, 255]` depending
            on how your preprocessing pipeline is setup.
    """

    def __init__(
        self,
        value_range,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.value_range = value_range

    def augment_image(self, image, transformation=None, **kwargs):
        original_image = image
        image = preprocessing.transform_value_range(
            image,
            original_range=self.value_range,
            target_range=(0, 255),
            dtype=self.compute_dtype,
        )

        low = tf.reduce_min(tf.reduce_min(image, axis=0), axis=0)
        high = tf.reduce_max(tf.reduce_max(image, axis=0), axis=0)
        scale = 255.0 / (high - low)
        offset = -low * scale

        image = image * scale[None, None] + offset[None, None]
        result = tf.clip_by_value(image, 0.0, 255.0)
        result = preprocessing.transform_value_range(
            result,
            original_range=(0, 255),
            target_range=self.value_range,
            dtype=self.compute_dtype,
        )
        # don't process NaN channels
        result = tf.where(tf.math.is_nan(result), original_image, result)
        return result

    def augment_bounding_boxes(self, bounding_boxes, **kwargs):
        return bounding_boxes

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def augment_segmentation_mask(self, segmentation_mask, transformation, **kwargs):
        return segmentation_mask

    def get_config(self):
        config = super().get_config()
        config.update({"value_range": self.value_range})
        return config


class AutoContrastConsistencyTest(tf.test.TestCase):
    def test_consistency_with_old_implementation(self):
        images = tf.random.uniform(shape=(16, 32, 32, 3))

        output = AutoContrast(value_range=(0, 1))(images)
        old_output = OldAutoContrast(value_range=(0, 1))(images)

        self.assertAllClose(old_output, output)


if __name__ == "__main__":
    (x_train, _), _ = keras.datasets.cifar10.load_data()
    x_train = x_train.astype(float)

    images = []
    num_images = [1000, 2000, 5000, 10000]
    results = {}

    for aug in [AutoContrast, OldAutoContrast]:
        c = aug.__name__

        layer = aug(value_range=(0, 255))

        runtimes = []
        print(f"Timing {c}")

        for n_images in num_images:
            # warmup
            layer(x_train[:n_images])

            t0 = time.time()
            r1 = layer(x_train[:n_images])
            t1 = time.time()
            runtimes.append(t1 - t0)
            print(f"Runtime for {c}, n_images={n_images}: {t1 - t0}")

        results[c] = runtimes

        c = aug.__name__ + " Graph Mode"

        layer = aug(value_range=(0, 255))

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
            print(f"Runtime for {c}, n_images={n_images}: {t1 - t0}")

        results[c] = runtimes

    plt.figure()
    for key in results:
        plt.plot(num_images, results[key], label=key)
        plt.xlabel("Number images")

    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.show()

    # So we can actually see more relevant margins
    del results["OldAutoContrast"]

    plt.figure()
    for key in results:
        plt.plot(num_images, results[key], label=key)
        plt.xlabel("Number images")

    plt.ylabel("Runtime (seconds)")
    plt.legend()
    plt.show()

    # Compare two implementations
    tf.test.main()
