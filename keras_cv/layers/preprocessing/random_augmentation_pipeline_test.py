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
from absl.testing import parameterized

from keras_cv import layers


class CountInvocations(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = tf.Variable(0, trainable=False)

    def call(self, inputs):
        self.calls.assign_add(1)
        return inputs


class RandomAugmentationPipelineTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(("1", 1), ("3", 3), ("5", 5))
    def test_calls_layers_augmentations_per_image_times(self, augmentations_per_image):
        layer = CountInvocations()
        pipeline = layers.RandomAugmentationPipeline(
            layers=[layer], augmentations_per_image=augmentations_per_image
        )
        xs = tf.random.uniform((2, 512, 512, 3), 0, 255, dtype=tf.float32)
        os = pipeline(xs)
        self.assertAllClose(xs, os)
        self.assertEqual(layer.calls, 2 * augmentations_per_image)

    @parameterized.named_parameters(("1", 1), ("3", 3), ("5", 5))
    def test_calls_layers_augmentations_per_image_times_single_image(
        self, augmentations_per_image
    ):
        layer = CountInvocations()
        pipeline = layers.RandomAugmentationPipeline(
            layers=[layer], augmentations_per_image=augmentations_per_image, rate=1.0
        )
        xs = tf.random.uniform((512, 512, 3), 0, 255, dtype=tf.float32)
        os = pipeline(xs)
        self.assertAllClose(xs, os)
        self.assertEqual(layer.calls, augmentations_per_image)
