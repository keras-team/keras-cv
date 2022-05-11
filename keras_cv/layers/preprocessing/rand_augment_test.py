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


class RandAugmentTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("0", 0),
        ("20", 0.2),
        ("55", 0.55),
        ("10", 1.0),
    )
    def test_runs_with_magnitude(self, magnitude):
        rand_augment = layers.RandAugment(
            value_range=(0, 255), rate=0.5, magnitude=magnitude
        )
        xs = tf.ones((2, 512, 512, 3))
        ys = rand_augment(xs)
        self.assertEqual(ys.shape, (2, 512, 512, 3))

    @parameterized.named_parameters(
        ("0_255", 0, 255),
        ("neg_1_1", -1, 1),
        ("0_1", 0, 1),
    )
    def test_runs_with_value_range(self, low, high):
        rand_augment = layers.RandAugment(
            augmentations_per_image=3, magnitude=0.5, rate=1.0, value_range=(low, high)
        )
        xs = tf.random.uniform((2, 512, 512, 3), low, high, dtype=tf.float32)
        ys = rand_augment(xs)
        self.assertTrue(tf.math.reduce_all(tf.logical_and(ys >= low, ys <= high)))

    @parameterized.named_parameters(
        ("float32", tf.float32),
        ("int32", tf.int32),
        ("uint8", tf.uint8),
    )
    def test_runs_with_dtype_input(self, dtype):
        rand_augment = layers.RandAugment(value_range=(0, 255))
        xs = tf.ones((2, 512, 512, 3), dtype=dtype)
        ys = rand_augment(xs)
        self.assertEqual(ys.shape, (2, 512, 512, 3))

    @parameterized.named_parameters(
        ("0_255", 0, 255),
        ("neg1_1", -1, 1),
        ("0_1", 0, 1),
    )
    def test_standard_policy_respects_value_range(self, lower, upper):
        my_layers = layers.RandAugment.get_standard_policy(
            value_range=(lower, upper), magnitude=1.0, magnitude_stddev=0.2
        )
        rand_augment = layers.RandomAugmentationPipeline(
            layers=my_layers, augmentations_per_image=3
        )
        xs = tf.random.uniform((2, 512, 512, 3), lower, upper, dtype=tf.float32)
        ys = rand_augment(xs)
        self.assertLessEqual(tf.math.reduce_max(ys), upper)
        self.assertGreaterEqual(tf.math.reduce_min(ys), lower)

    def test_runs_unbatched(self):
        rand_augment = layers.RandAugment(
            augmentations_per_image=3, magnitude=0.5, rate=1.0, value_range=(0, 255)
        )
        xs = tf.random.uniform((512, 512, 3), 0, 255, dtype=tf.float32)
        ys = rand_augment(xs)
        self.assertEqual(xs.shape, ys.shape)
