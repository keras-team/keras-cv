# Copyright 2022 The KerasCV Authors
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

import tensorflow as tf

from keras_cv import layers
from keras_cv.utils import preprocessing


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class AugMix(tf.keras.__internal__.layers.BaseImageAugmentationLayer):
    """Performs the AugMix data augmentation technique.  AugMix mixes several
    chains of augmentations together using weights sampled from a Dirichlet
    distribution.  A single chain consist of a series of individual augmentations.
    The resultant image is further mixed with the original image to form the
    final augmented image.

    Args:
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written [low, high].
            This is typically either `[0, 1]` or `[0, 255]` depending
            on how your preprocessing pipeline is setup.
        severity: A tuple of two floats, a single float or a `keras_cv.FactorSampler`.
            A value is sampled from the provided range.  If a float is passed, the
            range is interpreted as `(0, severity)`. This value represents the
            level of strength of augmentations.  Defaults to 0.3.
        width: an integer representing the number of different chains to
            be mixed. Defaults to 3.
        depth: an integer representing the number of transformations in
            the chains. A negative value enables stochastic depth uniformly
            in [1,3]. Defaults to -1.
        alpha: a float value used as the probability coefficients for the
            Beta and Dirichlet distributions. Defaults to 1.0.
        seed: Integer. Used to create a random seed.

    References:
        [AugMix paper](https://arxiv.org/pdf/1912.02781)
        [Official Code](https://github.com/google-research/augmix)
        [Unoffial TF Code](https://github.com/szacho/augmix-tf)

    Sample usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    augmix = keras_cv.layers.preprocessing.mix_up.AugMix([0, 255])
    augmented_images = augmix(images)
    ```
    """

    def __init__(
        self,
        value_range,
        severity=0.3,
        width=3,
        depth=-1,
        alpha=1.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.value_range = value_range
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.seed = seed
        self.auto_vectorize = False
        self.severity = severity

        # initialize layers
        self.auto_contrast = layers.AutoContrast(value_range = self.value_range)
        self.equalize = layers.Equalization(value_range = self.value_range)
        self.random_shear = layers.RandomShear

    @staticmethod
    def _sample_from_dirichlet(alpha):
        gamma_sample = tf.random.gamma(shape=(), alpha=alpha)
        return gamma_sample / tf.reduce_sum(gamma_sample, axis=-1, keepdims=True)

    @staticmethod
    def _sample_from_beta(alpha, beta):
        sample_alpha = tf.random.gamma((), 1.0, beta=alpha)
        sample_beta = tf.random.gamma((), 1.0, beta=beta)
        return sample_alpha / (sample_alpha + sample_beta)

    def _sample_level(self, level, maxval, dtype):
        level = (
            self._random_generator.random_uniform(
                shape=(), minval=0.01, maxval=level, dtype=tf.float32
            )
            * 10
        )

        return tf.cast((level) * maxval / 10.0, dtype)

    def _loop_on_depth(self, depth_level, image_aug):
        op_index = self._random_generator.random_uniform(
            shape=(), minval=0, maxval=8, dtype=tf.int32
        )
        image_aug = self._apply_op(image_aug, op_index)
        depth_level += 1
        return depth_level, image_aug

    def _loop_on_width(self, image, chain_mixing_weights, curr_chain, result):
        image_aug = tf.identity(image)
        depth = tf.cond(
            tf.greater(self.depth, 0),
            lambda: self.depth,
            lambda: self._random_generator.random_uniform(
                shape=(), minval=1, maxval=3, dtype=tf.int32
            ),
        )

        depth_level = tf.constant([0], dtype=tf.int32)
        depth_level, image_aug = tf.while_loop(
            lambda depth_level, image_aug: tf.less(depth_level, depth),
            self._loop_on_depth,
            [depth_level, image_aug],
        )
        result += tf.gather(chain_mixing_weights, curr_chain) * image_aug
        curr_chain += 1
        return image, chain_mixing_weights, curr_chain, result

    def _auto_contrast(self, image):
        return self.auto_contrast(image)

    def _equalize(self, image):
        return self.equalize(image)

    def _posterize(self, image):
        image = preprocessing.transform_value_range(
            images=image,
            original_range=self.value_range,
            target_range=[0, 255],
        )

        bits = self._sample_level(self.severity, 3, tf.int32)
        shift = tf.cast(4 - bits + 1, tf.uint8)
        image = tf.cast(image, tf.uint8)
        image = tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)
        image = tf.cast(image, self.compute_dtype)
        return preprocessing.transform_value_range(
            images=image,
            original_range=[0, 255],
            target_range=self.value_range,
        )

    def _rotate(self, image):
        angle = tf.expand_dims(
            self._sample_level(self.severity, 30, tf.float32), axis=0
        )
        shape = tf.cast(tf.shape(image), tf.float32)

        return preprocessing.transform(
            tf.expand_dims(image, 0),
            preprocessing.get_rotation_matrix(angle, shape[0], shape[1]),
        )[0]

    def _solarize(self, image):
        threshold = tf.cast(
            self._sample_level(self.severity, 255, tf.int32), tf.float32
        )

        image = preprocessing.transform_value_range(
            image, original_range=self.value_range, target_range=(0, 255)
        )
        result = tf.clip_by_value(image, 0, 255)
        result = tf.where(result < threshold, result, 255 - result)
        return preprocessing.transform_value_range(
            result, original_range=(0, 255), target_range=self.value_range
        )

    def _shear_x(self, image):
        x = self._sample_level(self.severity, 0.3, tf.float32)
        x *= preprocessing.random_inversion(self._random_generator)
        transform_x = self.random_shear._format_transform(
            [1.0, x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        )
        return preprocessing.transform(
            images=tf.expand_dims(image, 0), transforms=transform_x
        )[0]

    def _shear_y(self, image):
        y = self._sample_level(self.severity, 0.3, tf.float32)
        y *= preprocessing.random_inversion(self._random_generator)
        transform_x = self.random_shear._format_transform(
            [1.0, 0.0, 0.0, y, 1.0, 0.0, 0.0, 0.0]
        )
        return preprocessing.transform(
            images=tf.expand_dims(image, 0), transforms=transform_x
        )[0]

    def _translate_x(self, image):
        shape = tf.cast(tf.shape(image), tf.float32)
        x = self._sample_level(self.severity, shape[1] / 3, tf.float32)
        x = tf.expand_dims(tf.expand_dims(x, axis=0), axis=0)
        x *= preprocessing.random_inversion(self._random_generator)
        x = tf.cast(x, tf.int32)

        translations = tf.cast(
            tf.concat([x, tf.zeros_like(x)], axis=1), dtype=tf.float32
        )
        return preprocessing.transform(
            tf.expand_dims(image, 0), preprocessing.get_translation_matrix(translations)
        )[0]

    def _translate_y(self, image):
        shape = tf.cast(tf.shape(image), tf.float32)
        y = self._sample_level(self.severity, shape[0] / 3, tf.float32)
        y = tf.expand_dims(tf.expand_dims(y, axis=0), axis=0)
        y *= preprocessing.random_inversion(self._random_generator)
        y = tf.cast(y, tf.int32)

        translations = tf.cast(
            tf.concat([tf.zeros_like(y), y], axis=1), dtype=tf.float32
        )
        return preprocessing.transform(
            tf.expand_dims(image, 0), preprocessing.get_translation_matrix(translations)
        )[0]

    def _apply_op(self, image, op_index):
        augmented = image
        augmented = tf.cond(
            op_index == tf.constant([0], dtype=tf.int32),
            lambda: self._auto_contrast(augmented),
            lambda: augmented,
        )
        augmented = tf.cond(
            op_index == tf.constant([1], dtype=tf.int32),
            lambda: self._equalize(augmented),
            lambda: augmented,
        )
        augmented = tf.cond(
            op_index == tf.constant([2], dtype=tf.int32),
            lambda: self._posterize(augmented),
            lambda: augmented,
        )
        augmented = tf.cond(
            op_index == tf.constant([3], dtype=tf.int32),
            lambda: self._rotate(augmented),
            lambda: augmented,
        )
        augmented = tf.cond(
            op_index == tf.constant([4], dtype=tf.int32),
            lambda: self._solarize(augmented),
            lambda: augmented,
        )
        augmented = tf.cond(
            op_index == tf.constant([5], dtype=tf.int32),
            lambda: self._shear_x(augmented),
            lambda: augmented,
        )
        augmented = tf.cond(
            op_index == tf.constant([6], dtype=tf.int32),
            lambda: self._shear_y(augmented),
            lambda: augmented,
        )
        augmented = tf.cond(
            op_index == tf.constant([7], dtype=tf.int32),
            lambda: self._translate_x(augmented),
            lambda: augmented,
        )
        augmented = tf.cond(
            op_index == tf.constant([8], dtype=tf.int32),
            lambda: self._translate_y(augmented),
            lambda: augmented,
        )
        return augmented

    def augment_image(self, image, transformation=None):
        chain_mixing_weights = AugMix._sample_from_dirichlet(
            tf.ones([self.width]) * self.alpha
        )
        weight_sample = AugMix._sample_from_beta(self.alpha, self.alpha)

        result = tf.zeros_like(image)
        curr_chain = tf.constant([0], dtype=tf.int32)

        image, chain_mixing_weights, curr_chain, result = tf.while_loop(
            lambda image, chain_mixing_weights, curr_chain, result: tf.less(
                curr_chain, self.width
            ),
            self._loop_on_width,
            [image, chain_mixing_weights, curr_chain, result],
        )
        result = weight_sample * image + (1 - weight_sample) * result
        return result

    def augment_label(self, label, transformation=None):
        return label

    def get_config(self):
        config = {
            "value_range": self.value_range,
            "severity": self.severity,
            "width": self.width,
            "depth": self.depth,
            "alpha": self.alpha,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
