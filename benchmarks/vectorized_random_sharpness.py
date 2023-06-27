import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

from keras_cv.layers import BaseImageAugmentationLayer
from keras_cv.layers import RandomSharpness
from keras_cv.utils import preprocessing


class OldRandomSharpness(BaseImageAugmentationLayer):
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

    def get_random_transformation(self, **kwargs):
        return self.factor(dtype=self.compute_dtype)

    def augment_image(self, image, transformation=None, **kwargs):
        image = preprocessing.transform_value_range(
            image,
            original_range=self.value_range,
            target_range=(0, 255),
            dtype=self.compute_dtype,
        )
        original_image = image

        # Make image 4D for conv operation.
        image = tf.expand_dims(image, axis=0)

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
        channels = tf.shape(image)[-1]
        kernel = tf.tile(kernel, [1, 1, channels, 1])
        strides = [1, 1, 1, 1]

        smoothed_image = tf.nn.depthwise_conv2d(
            image, kernel, strides, padding="VALID", dilations=[1, 1]
        )
        smoothed_image = tf.clip_by_value(smoothed_image, 0.0, 255.0)
        smoothed_image = tf.squeeze(smoothed_image, axis=0)

        # For the borders of the resulting image, fill in the values of the
        # original image.
        mask = tf.ones_like(smoothed_image)
        padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
        padded_smoothed_image = tf.pad(smoothed_image, [[1, 1], [1, 1], [0, 0]])

        result = tf.where(
            tf.equal(padded_mask, 1), padded_smoothed_image, original_image
        )
        # Blend the final result.
        result = preprocessing.blend(original_image, result, transformation)
        result = preprocessing.transform_value_range(
            result,
            original_range=(0, 255),
            target_range=self.value_range,
            dtype=self.compute_dtype,
        )
        return result

    def augment_bounding_boxes(self, bounding_boxes, transformation, **kwargs):
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
                "factor": self.factor,
                "value_range": self.value_range,
                "seed": self.seed,
            }
        )
        return config


class RandomSharpnessTest(tf.test.TestCase):
    def test_consistency_with_old_implementation(self):
        images = tf.random.uniform(shape=(2, 64, 64, 3), minval=0, maxval=255)

        old_layer = OldRandomSharpness(value_range=(0, 255), factor=(0.5, 0.5))
        new_layer = RandomSharpness(value_range=(0, 255), factor=(0.5, 0.5))

        old_output = old_layer(images)
        new_output = new_layer(images)

        self.assertAllClose(old_output, new_output)


if __name__ == "__main__":
    # Run benchmark
    (x_train, _), _ = keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32)

    num_images = [1000, 2000, 5000, 10000]
    results = {}
    aug_candidates = [RandomSharpness, OldRandomSharpness]
    aug_args = {"value_range": (0, 255), "factor": 0.5}

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
