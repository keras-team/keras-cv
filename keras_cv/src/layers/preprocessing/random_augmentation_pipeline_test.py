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

import pytest

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "To use KerasCV, please install TensorFlow: `pip install tensorflow`. "
        "The TensorFlow package is required for data preprocessing with any backend."
    )
from absl.testing import parameterized

from keras_cv.src import layers
from keras_cv.src.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.src.tests.test_case import TestCase


class AddOneToInputs(BaseImageAugmentationLayer):
    """Add 1 to all image values, for testing purposes."""

    def __init__(self):
        super(AddOneToInputs, self).__init__()

    def augment_image(self, image, transformation=None, **kwargs):
        return image + 1


class RandomAugmentationPipelineTest(TestCase):
    @parameterized.named_parameters(("1", 1), ("3", 3), ("5", 5))
    def test_calls_layers_augmentations_per_image_times(
        self, augmentations_per_image
    ):
        layer = AddOneToInputs()
        pipeline = layers.RandomAugmentationPipeline(
            layers=[layer],
            augmentations_per_image=augmentations_per_image,
            rate=1.0,
        )
        xs = tf.random.uniform((2, 5, 5, 3), 0, 100, dtype=tf.float32)
        os = pipeline(xs)

        self.assertAllClose(xs + augmentations_per_image, os)

    def test_supports_empty_layers_argument(self):
        pipeline = layers.RandomAugmentationPipeline(
            layers=[], augmentations_per_image=1, rate=1.0
        )
        xs = tf.random.uniform((2, 5, 5, 3), 0, 100, dtype=tf.float32)
        os = pipeline(xs)

        self.assertAllClose(xs, os)

    @pytest.mark.tf_keras_only
    def test_calls_layers_augmentations_in_graph(self):
        layer = AddOneToInputs()
        pipeline = layers.RandomAugmentationPipeline(
            layers=[layer], augmentations_per_image=3, rate=1.0
        )

        @tf.function()
        def call_pipeline(xs):
            return pipeline(xs)

        xs = tf.random.uniform((2, 5, 5, 3), 0, 100, dtype=tf.float32)
        os = call_pipeline(xs)

        self.assertAllClose(xs + 3, os)

    @parameterized.named_parameters(("1", 1), ("3", 3), ("5", 5))
    def test_calls_layers_augmentations_per_image_times_single_image(
        self, augmentations_per_image
    ):
        layer = AddOneToInputs()
        pipeline = layers.RandomAugmentationPipeline(
            layers=[layer],
            augmentations_per_image=augmentations_per_image,
            rate=1.0,
        )
        xs = tf.random.uniform((5, 5, 3), 0, 100, dtype=tf.float32)
        os = pipeline(xs)

        self.assertAllClose(xs + augmentations_per_image, os)

    @parameterized.named_parameters(("1", 1), ("3", 3), ("5", 5))
    def test_respects_rate(self, augmentations_per_image):
        layer = AddOneToInputs()
        pipeline = layers.RandomAugmentationPipeline(
            layers=[layer],
            augmentations_per_image=augmentations_per_image,
            rate=0.0,
        )
        xs = tf.random.uniform((2, 5, 5, 3), 0, 100, dtype=tf.float32)
        os = pipeline(xs)

        self.assertAllClose(xs, os)
