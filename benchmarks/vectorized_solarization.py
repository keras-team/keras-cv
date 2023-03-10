import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from keras_cv.layers import BaseImageAugmentationLayer
from keras_cv.layers import Solarization
from keras_cv.utils import preprocessing


class OldSolarization(BaseImageAugmentationLayer):
    def __init__(
        self,
        value_range,
        addition_factor=0.0,
        threshold_factor=0.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.seed = seed
        self.addition_factor = preprocessing.parse_factor(
            addition_factor,
            max_value=255,
            seed=seed,
            param_name="addition_factor",
        )
        self.threshold_factor = preprocessing.parse_factor(
            threshold_factor,
            max_value=255,
            seed=seed,
            param_name="threshold_factor",
        )
        self.value_range = value_range

    def get_random_transformation(self, **kwargs):
        return (
            self.addition_factor(dtype=self.compute_dtype),
            self.threshold_factor(dtype=self.compute_dtype),
        )

    def augment_image(self, image, transformation=None, **kwargs):
        (addition, threshold) = transformation
        image = preprocessing.transform_value_range(
            image,
            original_range=self.value_range,
            target_range=(0, 255),
            dtype=self.compute_dtype,
        )
        result = image + addition
        result = tf.clip_by_value(result, 0, 255)
        result = tf.where(result < threshold, result, 255 - result)
        result = preprocessing.transform_value_range(
            result,
            original_range=(0, 255),
            target_range=self.value_range,
            dtype=self.compute_dtype,
        )
        return result

    def augment_bounding_boxes(self, bounding_boxes, **kwargs):
        return bounding_boxes

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def augment_segmentation_mask(
        self, segmentation_mask, transformation, **kwargs
    ):
        return segmentation_mask

    def get_config(self):
        config = {
            "threshold_factor": self.threshold_factor,
            "addition_factor": self.addition_factor,
            "value_range": self.value_range,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if isinstance(config["threshold_factor"], dict):
            config[
                "threshold_factor"
            ] = tf.keras.utils.deserialize_keras_object(
                config["threshold_factor"]
            )
        if isinstance(config["addition_factor"], dict):
            config["addition_factor"] = tf.keras.utils.deserialize_keras_object(
                config["addition_factor"]
            )
        return cls(**config)


class SolarizationTest(tf.test.TestCase):
    def test_consistency_with_old_implementation(self):
        images = tf.random.uniform(shape=(16, 32, 32, 3))

        output = Solarization(
            value_range=(0, 1),
            threshold_factor=(200, 200),
            addition_factor=(100, 100),
        )(images)
        old_output = OldSolarization(
            value_range=(0, 1),
            threshold_factor=(200, 200),
            addition_factor=(100, 100),
        )(images)

        self.assertAllClose(old_output, output)


if __name__ == "__main__":
    # Run benchmark
    (x_train, _), _ = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32)

    num_images = [1000, 2000, 5000, 10000]
    results = {}
    aug_candidates = [Solarization, OldSolarization]
    aug_args = {"value_range": (0, 255), "threshold_factor": 0.5}

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
