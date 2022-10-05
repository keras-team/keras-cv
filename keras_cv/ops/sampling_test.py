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

from keras_cv.ops.sampling import balanced_sample


class BalancedSamplingTest(tf.test.TestCase):
    def test_balanced_sampling(self):
        positive_matches = tf.constant(
            [True, False, False, False, False, False, False, False, False, False]
        )
        negative_matches = tf.constant(
            [False, True, True, True, True, True, True, True, True, True]
        )
        num_samples = 5
        positive_fraction = 0.2
        res = balanced_sample(
            positive_matches, negative_matches, num_samples, positive_fraction
        )
        # The 1st element must be selected, given it's the only one.
        self.assertAllClose(res[0], 1)

    def test_balanced_batched_sampling(self):
        positive_matches = tf.constant(
            [
                [True, False, False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, True, False, False, False],
            ]
        )
        negative_matches = tf.constant(
            [
                [False, True, True, True, True, True, True, True, True, True],
                [True, True, True, True, True, True, False, True, True, True],
            ]
        )
        num_samples = 5
        positive_fraction = 0.2
        res = balanced_sample(
            positive_matches, negative_matches, num_samples, positive_fraction
        )
        # the 1st element from the 1st batch must be selected, given it's the only one
        self.assertAllClose(res[0][0], 1)
        # the 7th element from the 2nd batch must be selected, given it's the only one
        self.assertAllClose(res[1][6], 1)

    def test_balanced_sampling_over_positive_fraction(self):
        positive_matches = tf.constant(
            [True, False, False, False, False, False, False, False, False, False]
        )
        negative_matches = tf.constant(
            [False, True, True, True, True, True, True, True, True, True]
        )
        num_samples = 5
        positive_fraction = 0.4
        res = balanced_sample(
            positive_matches, negative_matches, num_samples, positive_fraction
        )
        # only 1 positive sample exists, thus it is chosen
        self.assertAllClose(res[0], 1)

    def test_balanced_sampling_under_positive_fraction(self):
        positive_matches = tf.constant(
            [True, False, False, False, False, False, False, False, False, False]
        )
        negative_matches = tf.constant(
            [False, True, True, True, True, True, True, True, True, True]
        )
        num_samples = 5
        positive_fraction = 0.1
        res = balanced_sample(
            positive_matches, negative_matches, num_samples, positive_fraction
        )
        # no positive is chosen
        self.assertAllClose(res[0], 0)
        self.assertAllClose(tf.reduce_sum(res), 5)

    def test_balanced_sampling_over_num_samples(self):
        positive_matches = tf.constant(
            [True, False, False, False, False, False, False, False, False, False]
        )
        negative_matches = tf.constant(
            [False, True, True, True, True, True, True, True, True, True]
        )
        # users want to get 20 samples, but only 10 are available
        num_samples = 20
        positive_fraction = 0.1
        with self.assertRaisesRegex(ValueError, "has less element"):
            _ = balanced_sample(
                positive_matches, negative_matches, num_samples, positive_fraction
            )

    def test_balanced_sampling_no_positive(self):
        positive_matches = tf.constant(
            [False, False, False, False, False, False, False, False, False, False]
        )
        # the rest are neither positive nor negative, but ignord matches
        negative_matches = tf.constant(
            [False, False, True, False, False, True, False, False, True, False]
        )
        num_samples = 5
        positive_fraction = 0.5
        res = balanced_sample(
            positive_matches, negative_matches, num_samples, positive_fraction
        )
        # given only 3 negative and 0 positive, select all of them
        self.assertAllClose(res, [0, 0, 1, 0, 0, 1, 0, 0, 1, 0])

    def test_balanced_sampling_no_negative(self):
        positive_matches = tf.constant(
            [True, True, False, False, False, False, False, False, False, False]
        )
        # 2-9 indices are neither positive nor negative, they're ignored matches
        negative_matches = tf.constant([False] * 10)
        num_samples = 5
        positive_fraction = 0.5
        res = balanced_sample(
            positive_matches, negative_matches, num_samples, positive_fraction
        )
        # given only 2 positive and 0 negative, select all of them.
        self.assertAllClose(res, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_balanced_sampling_many_samples(self):
        positive_matches = tf.random.uniform(
            [2, 1000], minval=0, maxval=1, dtype=tf.float32
        )
        positive_matches = positive_matches > 0.98
        negative_matches = tf.logical_not(positive_matches)
        num_samples = 256
        positive_fraction = 0.25
        _ = balanced_sample(
            positive_matches, negative_matches, num_samples, positive_fraction
        )
