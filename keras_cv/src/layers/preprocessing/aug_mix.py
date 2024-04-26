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

from keras_cv.src import layers
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.src.utils import preprocessing


@keras_cv_export("keras_cv.layers.AugMix")
class AugMix(BaseImageAugmentationLayer):
    """Performs the AugMix data augmentation technique.

    AugMix aims to produce images with variety while preserving the image
    semantics and local statistics. During the augmentation process, each image
    is augmented `num_chains` different ways, each way consisting of
    `chain_depth` augmentations. Augmentations are sampled from the list:
    translation, shearing, rotation, posterization, histogram equalization,
    solarization and auto contrast. The results of each chain are then mixed
    together with the original image based on random samples from a Dirichlet
    distribution.

    Args:
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written (low, high).
            This is typically either `(0, 1)` or `(0, 255)` depending
            on how your preprocessing pipeline is set up.
        severity: A tuple of two floats, a single float or a
            `keras_cv.FactorSampler`. A value is sampled from the provided
            range. If a float is passed, the range is interpreted as
            `(0, severity)`. This value represents the level of strength of
            augmentations and is in the range [0, 1]. Defaults to 0.3.
        num_chains: an integer representing the number of different chains to
            be mixed, defaults to 3.
        chain_depth: an integer or range representing the number of
            transformations in the chains. If a range is passed, a random
            `chain_depth` value sampled from a uniform distribution over the
            given range is called at the start of the chain. Defaults to [1,3].
        alpha: a float value used as the probability coefficients for the
            Beta and Dirichlet distributions, defaults to 1.0.
        seed: Integer. Used to create a random seed.

    References:
        - [AugMix paper](https://arxiv.org/pdf/1912.02781)
        - [Official Code](https://github.com/google-research/augmix)
        - [Unofficial TF Code](https://github.com/szacho/augmix-tf)

    Example:
    ```python
    (images, labels), _ = keras.datasets.cifar10.load_data()
    augmix = keras_cv.layers.AugMix([0, 255])
    augmented_images = augmix(images[:100])
    ```
    """

    def __init__(
        self,
        value_range,
        severity=0.3,
        num_chains=3,
        chain_depth=[1, 3],
        alpha=1.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.value_range = value_range
        self.num_chains = num_chains
        self.chain_depth = chain_depth

        if isinstance(self.chain_depth, int):
            self.chain_depth = [self.chain_depth, self.chain_depth]

        self.alpha = alpha
        self.seed = seed
        self.auto_vectorize = False
        self.severity = severity
        self.severity_factor = preprocessing.parse_factor(
            self.severity,
            min_value=0.01,
            max_value=1.0,
            param_name="severity",
            seed=self.seed,
        )

        # initialize layers
        self.auto_contrast = layers.AutoContrast(value_range=self.value_range)
        self.equalize = layers.Equalization(value_range=self.value_range)

    def _sample_from_dirichlet(self, alpha):
        gamma_sample = tf.random.gamma(
            shape=(),
            alpha=alpha,
        )
        return gamma_sample / tf.reduce_sum(
            gamma_sample, axis=-1, keepdims=True
        )

    def _sample_from_beta(self, alpha, beta):
        sample_alpha = tf.random.gamma(
            (),
            alpha=alpha,
        )
        sample_beta = tf.random.gamma(
            (),
            alpha=beta,
        )
        return sample_alpha / (sample_alpha + sample_beta)

    def _sample_depth(self):
        return self._random_generator.uniform(
            shape=(),
            minval=self.chain_depth[0],
            maxval=self.chain_depth[1] + 1,
            dtype=tf.int32,
        )

    def _loop_on_depth(self, depth_level, image_aug):
        op_index = self._random_generator.uniform(
            shape=(), minval=0, maxval=8, dtype=tf.int32
        )
        image_aug = self._apply_op(image_aug, op_index)
        depth_level += 1
        return depth_level, image_aug

    def _loop_on_width(self, image, chain_mixing_weights, curr_chain, result):
        image_aug = tf.identity(image)
        chain_depth = self._sample_depth()

        depth_level = tf.constant([0], dtype=tf.int32)
        depth_level, image_aug = tf.while_loop(
            lambda depth_level, image_aug: tf.less(depth_level, chain_depth),
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

        bits = tf.cast(self.severity_factor() * 3, tf.int32)
        shift = tf.cast(4 - bits + 1, tf.uint8)
        image = tf.cast(image, tf.uint8)
        image = tf.bitwise.left_shift(
            tf.bitwise.right_shift(image, shift), shift
        )
        image = tf.cast(image, self.compute_dtype)
        return preprocessing.transform_value_range(
            images=image,
            original_range=[0, 255],
            target_range=self.value_range,
        )

    def _rotate(self, image):
        angle = tf.expand_dims(
            tf.cast(self.severity_factor() * 30, tf.float32), axis=0
        )
        shape = tf.cast(tf.shape(image), tf.float32)

        return preprocessing.transform(
            tf.expand_dims(image, 0),
            preprocessing.get_rotation_matrix(angle, shape[0], shape[1]),
        )[0]

    def _solarize(self, image):
        threshold = tf.cast(
            tf.cast(self.severity_factor() * 255, tf.int32), tf.float32
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
        x = tf.cast(self.severity_factor() * 0.3, tf.float32)
        x *= preprocessing.random_inversion(self._random_generator)
        transform_x = layers.RandomShear._format_transform(
            [1.0, x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        )
        return preprocessing.transform(
            images=tf.expand_dims(image, 0), transforms=transform_x
        )[0]

    def _shear_y(self, image):
        y = tf.cast(self.severity_factor() * 0.3, tf.float32)
        y *= preprocessing.random_inversion(self._random_generator)
        transform_x = self._format_random_shear_transform(
            [1.0, 0.0, 0.0, y, 1.0, 0.0, 0.0, 0.0]
        )
        return preprocessing.transform(
            images=tf.expand_dims(image, 0), transforms=transform_x
        )[0]

    @staticmethod
    def _format_random_shear_transform(transform):
        transform = tf.convert_to_tensor(transform, dtype=tf.float32)
        return transform[tf.newaxis]

    def _translate_x(self, image):
        shape = tf.cast(tf.shape(image), tf.float32)
        x = tf.cast(self.severity_factor() * shape[1] / 3, tf.float32)
        x = tf.expand_dims(tf.expand_dims(x, axis=0), axis=0)
        x *= preprocessing.random_inversion(self._random_generator)
        x = tf.cast(x, tf.int32)

        translations = tf.cast(
            tf.concat([x, tf.zeros_like(x)], axis=1), dtype=tf.float32
        )
        return preprocessing.transform(
            tf.expand_dims(image, 0),
            preprocessing.get_translation_matrix(translations),
        )[0]

    def _translate_y(self, image):
        shape = tf.cast(tf.shape(image), tf.float32)
        y = tf.cast(self.severity_factor() * shape[0] / 3, tf.float32)
        y = tf.expand_dims(tf.expand_dims(y, axis=0), axis=0)
        y *= preprocessing.random_inversion(self._random_generator)
        y = tf.cast(y, tf.int32)

        translations = tf.cast(
            tf.concat([tf.zeros_like(y), y], axis=1), dtype=tf.float32
        )
        return preprocessing.transform(
            tf.expand_dims(image, 0),
            preprocessing.get_translation_matrix(translations),
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

    def get_random_transformation(
        self,
        image=None,
        label=None,
        bounding_boxes=None,
        keypoints=None,
        segmentation_mask=None,
    ):
        # Generate random values of chain_mixing_weights and weight_sample
        chain_mixing_weights = self._sample_from_dirichlet(
            tf.ones([self.num_chains]) * self.alpha
        )
        weight_sample = self._sample_from_beta(self.alpha, self.alpha)

        # Create a transformation config containing the random values
        transformation = {
            "chain_mixing_weights": chain_mixing_weights,
            "weight_sample": weight_sample,
        }

        return transformation

    def augment_image(self, image, transformation=None, **kwargs):
        # Extract chain_mixing_weights and weight_sample from the provided transformation # noqa: E501
        chain_mixing_weights = transformation["chain_mixing_weights"]
        weight_sample = transformation["weight_sample"]

        result = tf.zeros_like(image)
        curr_chain = tf.constant([0], dtype=tf.int32)

        image, chain_mixing_weights, curr_chain, result = tf.while_loop(
            lambda image, chain_mixing_weights, curr_chain, result: tf.less(
                curr_chain, self.num_chains
            ),
            self._loop_on_width,
            [image, chain_mixing_weights, curr_chain, result],
        )
        result = weight_sample * image + (1 - weight_sample) * result
        return result

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def augment_segmentation_mask(
        self, segmentation_masks, transformation=None, **kwargs
    ):
        # Extract chain_mixing_weights and weight_sample from the provided transformation # noqa: E501
        chain_mixing_weights = transformation["chain_mixing_weights"]
        weight_sample = transformation["weight_sample"]

        result = tf.zeros_like(segmentation_masks)
        curr_chain = tf.constant([0], dtype=tf.int32)

        (
            segmentation_masks,
            chain_mixing_weights,
            curr_chain,
            result,
        ) = tf.while_loop(
            lambda segmentation_masks, chain_mixing_weights, curr_chain, result: tf.less(  # noqa: E501
                curr_chain, self.num_chains
            ),
            self._loop_on_width,
            [segmentation_masks, chain_mixing_weights, curr_chain, result],
        )

        # Apply the mixing of segmentation_masks similar to images
        result = (
            weight_sample * segmentation_masks + (1 - weight_sample) * result
        )
        return result

    def get_config(self):
        config = {
            "value_range": self.value_range,
            "severity": self.severity,
            "num_chains": self.num_chains,
            "chain_depth": self.chain_depth,
            "alpha": self.alpha,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
