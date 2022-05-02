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
    """Performs the AugMix data augmentation technique.

    Args:
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written [low, high].
            This is typically either `[0, 1]` or `[0, 255]` depending
            on how your preprocessing pipeline is setup.
        severity: an integer representing the level of strength of
            augmentations (between 1 and 10). Defaults to 3.
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
        severity=3,
        width=3,
        depth=-1,
        alpha=1.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.value_range = value_range
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.seed = seed
        self.augmentations = [
            self._auto_contrast,
            self._equalize,
            self._posterize,
            self._rotate,
            self._solarize,
            self._shear_x,
            self._shear_y,
            self._translate_x,
            self._translate_y,
        ]

        self.auto_vectorize = False

    @staticmethod
    def _sample_from_dirichlet(alpha):
        gamma_sample = tf.random.gamma(shape=(), alpha=alpha)
        return gamma_sample / tf.reduce_sum(gamma_sample, -1, keepdims=True)

    @staticmethod
    def _sample_from_beta(alpha, beta):
        sample_alpha = tf.random.gamma((), 1.0, beta=alpha)
        sample_beta = tf.random.gamma((), 1.0, beta=beta)
        return sample_alpha / (sample_alpha + sample_beta)

    def _sample_level(self, level, maxval, dtype):
        level = self._random_generator.random_uniform(
            shape=(), minval=0.1, maxval=level, dtype=tf.float32
        )
        return tf.cast((level) * maxval / 10.0, dtype)

    def _loop_on_depth(self, depth_level, image_aug):
        op_index = tf.get_static_value(
            self._random_generator.random_uniform(
                shape=(), minval=0, maxval=8, dtype=tf.int32
            )
        )
        tf.print(op_index)
        image_aug = self.augmentations[op_index](image_aug)
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
        return layers.AutoContrast(self.value_range)(image)

    def _equalize(self, image):
        return layers.Equalization(self.value_range)(image)

    def _posterize(self, image):
        bits = self._sample_level(self.severity, 8, tf.int32)
        bits = tf.get_static_value(tf.math.maximum(bits, 1))
        return layers.Posterization(bits=bits, value_range=self.value_range)(image)

    def _rotate(self, image):
        angle = self.sample_level(self.severity, 30, tf.float32)
        shape = tf.shape(image, tf.float32)

        return preprocessing.transform(
            tf.expand_dims(image, 0),
            preprocessing.get_rotation_matrix(angle, shape[0], shape[1]),
        )[0]

    def _solarize(self, image):
        threshold = tf.get_static_value(
            self._sample_level(self.severity, 1, tf.float32)
        )
        return layers.Solarization(self.value_range, threshold_factor=(0, threshold))(
            image
        )

    def _shear_x(self, image):
        x = self._sample_level(self.severity, 0.3, tf.float32)
        x *= preprocessing.random_inversion(self._random_generator)
        transform_x = layers.RandomShear._format_transform(
            [1.0, x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        )
        return preprocessing.transform(
            images=tf.expand_dims(image, 0), transforms=transform_x
        )[0]

    def _shear_y(self, image):
        y = self._sample_level(self.severity, 0.3, tf.float32)
        y *= preprocessing.random_inversion(self._random_generator)
        transform_x = layers.RandomShear._format_transform(
            [1.0, 0.0, 0.0, y, 1.0, 0.0, 0.0, 0.0]
        )
        return preprocessing.transform(
            images=tf.expand_dims(image, 0), transforms=transform_x
        )[0]

    def _translate_x(self, image):
        shape = tf.shape(image, tf.float32)
        x = self.sample_level(self.severity, shape[1], tf.int32)
        x *= preprocessing.random_inversion(self._random_generator)

        translations = tf.cast(tf.concat([x, 0], axis=1), dtype=tf.float32)
        return preprocessing.transform(
            tf.expand_dims(image, 0), preprocessing.get_translation_matrix(translations)
        )[0]

    def _translate_y(self, image):
        shape = tf.shape(image, tf.float32)
        y = self.sample_level(self.severity, shape[0], tf.int32)
        y *= preprocessing.random_inversion(self._random_generator)

        translations = tf.cast(tf.concat([0, y], axis=1), dtype=tf.float32)
        return preprocessing.transform(
            tf.expand_dims(image, 0), preprocessing.get_translation_matrix(translations)
        )[0]

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
