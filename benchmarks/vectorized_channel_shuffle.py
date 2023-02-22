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

from keras_cv.layers import ChannelShuffle
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)


class OldChannelShuffle(BaseImageAugmentationLayer):
    """Shuffle channels of an input image.

    Input shape:
        The expected images should be [0-255] pixel ranges.
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format

    Args:
        groups: Number of groups to divide the input channels. Default 3.
        seed: Integer. Used to create a random seed.

    Call arguments:
        inputs: Tensor representing images of shape
            `(batch_size, width, height, channels)`, with dtype tf.float32 / tf.uint8,
            ` or (width, height, channels)`, with dtype tf.float32 / tf.uint8
        training: A boolean argument that determines whether the call should be run
            in inference mode or training mode. Default: True.

    Usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    channel_shuffle = keras_cv.layers.ChannelShuffle()
    augmented_images = channel_shuffle(images)
    ```
    """

    def __init__(self, groups=3, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.groups = groups
        self.seed = seed

    def augment_image(self, image, transformation=None, **kwargs):
        shape = tf.shape(image)
        height, width = shape[0], shape[1]
        num_channels = image.shape[2]

        if not num_channels % self.groups == 0:
            raise ValueError(
                "The number of input channels should be "
                "divisible by the number of groups."
                f"Received: channels={num_channels}, groups={self.groups}"
            )

        channels_per_group = num_channels // self.groups
        image = tf.reshape(
            image, [height, width, self.groups, channels_per_group]
        )
        image = tf.transpose(image, perm=[2, 0, 1, 3])
        image = tf.random.shuffle(image, seed=self.seed)
        image = tf.transpose(image, perm=[1, 2, 3, 0])
        image = tf.reshape(image, [height, width, num_channels])

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
        config.update({"groups": self.groups, "seed": self.seed})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class ChannelShuffleTest(tf.test.TestCase):
    def test_consistency_with_old_impl(self):
        # must set batch_size=1 due to randomness from
        # images = tf.random.shuffle(images, seed=self.seed)
        image_shape = (1, 32, 32, 3)
        fixed_seed = 2023
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = ChannelShuffle(groups=3, seed=fixed_seed)
        old_layer = OldChannelShuffle(groups=3, seed=fixed_seed)

        output = layer(image)
        old_output = old_layer(image)

        self.assertAllClose(old_output, output)


if __name__ == "__main__":
    # Run benchmark
    (x_train, _), _ = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32)

    num_images = [1000, 2000, 3000, 4000, 5000, 10000]
    results = {}
    aug_candidates = [ChannelShuffle, OldChannelShuffle]
    aug_args = {"groups": 3}

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
