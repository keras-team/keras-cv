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
from keras import backend
from tensorflow import keras

from keras_cv.layers import RandomTranslation
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.utils import preprocessing as preprocessing_utils

H_AXIS = -3
W_AXIS = -2


def check_fill_mode_and_interpolation(fill_mode, interpolation):
    if fill_mode not in {"reflect", "wrap", "constant", "nearest"}:
        raise NotImplementedError(
            f"Unknown `fill_mode` {fill_mode}. Only `reflect`, `wrap`, "
            "`constant` and `nearest` are supported."
        )
    if interpolation not in {"nearest", "bilinear"}:
        raise NotImplementedError(
            f"Unknown `interpolation` {interpolation}. Only `nearest` and "
            "`bilinear` are supported."
        )


def get_translation_matrix(translations, name=None):
    """Returns projective transform(s) for the given translation(s).

    Args:
      translations: A matrix of 2-element lists representing `[dx, dy]`
        to translate for each image (for a batch of images).
      name: The name of the op.

    Returns:
      A tensor of shape `(num_images, 8)` projective transforms which can be
        given to `transform`.
    """
    with backend.name_scope(name or "translation_matrix"):
        num_translations = tf.shape(translations)[0]
        # The translation matrix looks like:
        #     [[1 0 -dx]
        #      [0 1 -dy]
        #      [0 0 1]]
        # where the last entry is implicit.
        # Translation matrices are always float32.
        return tf.concat(
            values=[
                tf.ones((num_translations, 1), tf.float32),
                tf.zeros((num_translations, 1), tf.float32),
                -translations[:, 0, None],
                tf.zeros((num_translations, 1), tf.float32),
                tf.ones((num_translations, 1), tf.float32),
                -translations[:, 1, None],
                tf.zeros((num_translations, 2), tf.float32),
            ],
            axis=1,
        )


class OldRandomTranslation(BaseImageAugmentationLayer):
    """A preprocessing layer which randomly translates images during training.

    This layer will apply random translations to each image during training,
    filling empty space according to `fill_mode`.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype. By default, the layer will output
    floats.

    Args:
      height_factor: a float represented as fraction of value, or a tuple of
          size 2 representing lower and upper bound for shifting vertically. A
          negative value means shifting image up, while a positive value means
          shifting image down. When represented as a single positive float, this
          value is used for both the upper and lower bound. For instance,
          `height_factor=(-0.2, 0.3)` results in an output shifted by a random
          amount in the range `[-20%, +30%]`.  `height_factor=0.2` results in an
          output height shifted by a random amount in the range `[-20%, +20%]`.
      width_factor: a float represented as fraction of value, or a tuple of size
          2 representing lower and upper bound for shifting horizontally. A
          negative value means shifting image left, while a positive value means
          shifting image right. When represented as a single positive float,
          this value is used for both the upper and lower bound. For instance,
          `width_factor=(-0.2, 0.3)` results in an output shifted left by 20%,
          and shifted right by 30%. `width_factor=0.2` results
          in an output height shifted left or right by 20%.
      fill_mode: Points outside the boundaries of the input are filled according
          to the given mode
          (one of `{"constant", "reflect", "wrap", "nearest"}`).
          - *reflect*: `(d c b a | a b c d | d c b a)` The input is extended by
              reflecting about the edge of the last pixel.
          - *constant*: `(k k k k | a b c d | k k k k)` The input is extended by
              filling all values beyond the edge with the same constant value
              k = 0.
          - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
              wrapping around to the opposite edge.
          - *nearest*: `(a a a a | a b c d | d d d d)` The input is extended by
              the nearest pixel.
      interpolation: Interpolation mode. Supported values: `"nearest"`,
          `"bilinear"`.
      seed: Integer. Used to create a random seed.
      fill_value: a float represents the value to be filled outside the
          boundaries when `fill_mode="constant"`.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`,  in `"channels_last"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`,  in `"channels_last"` format.
    """

    def __init__(
        self,
        height_factor,
        width_factor,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        **kwargs,
    ):
        super().__init__(seed=seed, force_generator=True, **kwargs)
        self.height_factor = height_factor
        if isinstance(height_factor, (tuple, list)):
            self.height_lower = height_factor[0]
            self.height_upper = height_factor[1]
        else:
            self.height_lower = -height_factor
            self.height_upper = height_factor
        if self.height_upper < self.height_lower:
            raise ValueError(
                "`height_factor` cannot have upper bound less than "
                f"lower bound, got {height_factor}"
            )
        if abs(self.height_lower) > 1.0 or abs(self.height_upper) > 1.0:
            raise ValueError(
                "`height_factor` must have values between [-1, 1], "
                f"got {height_factor}"
            )

        self.width_factor = width_factor
        if isinstance(width_factor, (tuple, list)):
            self.width_lower = width_factor[0]
            self.width_upper = width_factor[1]
        else:
            self.width_lower = -width_factor
            self.width_upper = width_factor
        if self.width_upper < self.width_lower:
            raise ValueError(
                "`width_factor` cannot have upper bound less than "
                f"lower bound, got {width_factor}"
            )
        if abs(self.width_lower) > 1.0 or abs(self.width_upper) > 1.0:
            raise ValueError(
                "`width_factor` must have values between [-1, 1], "
                f"got {width_factor}"
            )

        check_fill_mode_and_interpolation(fill_mode, interpolation)

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.seed = seed

    def augment_image(self, image, transformation, **kwargs):
        """Translated inputs with random ops."""
        # The transform op only accepts rank 4 inputs, so if we have an
        # unbatched image, we need to temporarily expand dims to a batch.
        original_shape = image.shape
        inputs = tf.expand_dims(image, 0)

        inputs_shape = tf.shape(inputs)
        img_hd = tf.cast(inputs_shape[H_AXIS], tf.float32)
        img_wd = tf.cast(inputs_shape[W_AXIS], tf.float32)
        height_translation = transformation["height_translation"]
        width_translation = transformation["width_translation"]
        height_translation = height_translation * img_hd
        width_translation = width_translation * img_wd
        translations = tf.cast(
            tf.concat([width_translation, height_translation], axis=1),
            dtype=tf.float32,
        )
        output = preprocessing_utils.transform(
            inputs,
            get_translation_matrix(translations),
            interpolation=self.interpolation,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
        )

        output = tf.squeeze(output, 0)
        output.set_shape(original_shape)
        return output

    def get_random_transformation(self, image=None, **kwargs):
        batch_size = 1
        height_translation = self._random_generator.random_uniform(
            shape=[batch_size, 1],
            minval=self.height_lower,
            maxval=self.height_upper,
            dtype=tf.float32,
        )
        width_translation = self._random_generator.random_uniform(
            shape=[batch_size, 1],
            minval=self.width_lower,
            maxval=self.width_upper,
            dtype=tf.float32,
        )
        return {
            "height_translation": height_translation,
            "width_translation": width_translation,
        }

    def _batch_augment(self, inputs):
        # Change to vectorized_map for better performance, as well as work
        # around issue for different tensorspec between inputs and outputs.
        return tf.vectorized_map(self._augment, inputs)

    def augment_label(self, label, transformation, **kwargs):
        return label

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "height_factor": self.height_factor,
            "width_factor": self.width_factor,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "interpolation": self.interpolation,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RandomTranslationTest(tf.test.TestCase):
    def test_consistency_with_old_impl(self):
        image_shape = (16, 32, 32, 3)
        fixed_height_factor = (0.5, 0.5)
        fixed_width_factor = (0.5, 0.5)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = RandomTranslation(fixed_height_factor, fixed_width_factor)
        old_layer = OldRandomTranslation(
            fixed_height_factor, fixed_width_factor
        )

        output = layer(image)
        old_output = old_layer(image)

        self.assertAllClose(old_output, output)


if __name__ == "__main__":
    # Run benchmark
    (x_train, _), _ = keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32)

    num_images = [100, 200, 500, 1000]
    results = {}
    aug_candidates = [RandomTranslation, OldRandomTranslation]
    aug_args = {"height_factor": 0.5, "width_factor": 0.5}

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
        # cannot run tf.raw_ops.ImageProjectiveTransformV3 on XLA
        # for more information please refer:
        # https://github.com/tensorflow/tensorflow/issues/55194

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
