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

from keras_cv.layers import RandomSaturation
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.utils import preprocessing as preprocessing_utils


class OldRandomSaturation(BaseImageAugmentationLayer):
    """Randomly adjusts the saturation on given images.

    This layer will randomly increase/reduce the saturation for the input RGB
    images. At inference time, the output will be identical to the input.
    Call the layer with `training=True` to adjust the saturation of the input.

    Args:
        factor: A tuple of two floats, a single float or `keras_cv.FactorSampler`.
            `factor` controls the extent to which the image saturation is impacted.
            `factor=0.5` makes this layer perform a no-op operation. `factor=0.0` makes
            the image to be fully grayscale. `factor=1.0` makes the image to be fully
            saturated.
            Values should be between `0.0` and `1.0`. If a tuple is used, a `factor`
            is sampled between the two values for every image augmented.  If a single
            float is used, a value between `0.0` and the passed float is sampled.
            In order to ensure the value is always the same, please pass a tuple with
            two identical floats: `(0.5, 0.5)`.
        seed: Integer. Used to create a random seed.
    """

    def __init__(self, factor, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.factor = preprocessing_utils.parse_factor(
            factor,
            min_value=0.0,
            max_value=1.0,
        )
        self.seed = seed

    def get_random_transformation(self, **kwargs):
        return self.factor()

    def augment_image(self, image, transformation=None, **kwargs):
        # Convert the factor range from [0, 1] to [0, +inf]. Note that the
        # tf.image.adjust_saturation is trying to apply the following math formula
        # `output_saturation = input_saturation * factor`. We use the following
        # method to the do the mapping.
        # `y = x / (1 - x)`.
        # This will ensure:
        #   y = +inf when x = 1 (full saturation)
        #   y = 1 when x = 0.5 (no augmentation)
        #   y = 0 when x = 0 (full gray scale)

        # Convert the transformation to tensor in case it is a float. When
        # transformation is 1.0, then it will result in to divide by zero error, but
        # it will be handled correctly when it is a one tensor.
        transformation = tf.convert_to_tensor(transformation)
        adjust_factor = transformation / (1 - transformation)
        return tf.image.adjust_saturation(
            image, saturation_factor=adjust_factor
        )

    def augment_bounding_boxes(
        self, bounding_boxes, transformation=None, **kwargs
    ):
        return bounding_boxes

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def augment_segmentation_mask(
        self, segmentation_mask, transformation, **kwargs
    ):
        return segmentation_mask

    def get_config(self):
        config = {
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


(x_train, _), _ = keras.datasets.cifar10.load_data()
x_train = x_train.astype(np.float32)
x_train.shape

num_images = [1000, 2000, 5000, 10000]
results = {}
aug_candidates = [RandomSaturation, OldRandomSaturation]
aug_args = {"factor": (0.5)}

for aug in aug_candidates:
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
