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

from keras_cv.ops.box_matcher import ArgmaxBoxMatcher


class ArgmaxBoxMatcherTest(tf.test.TestCase):
    def test_box_matcher_unbatched(self):
        sim_matrix = tf.constant([[0.04, 0, 0, 0], [0, 0, 1.0, 0]], dtype=tf.float32)

        fg_threshold = 0.5
        bg_thresh_hi = 0.2
        bg_thresh_lo = 0.0

        matcher = ArgmaxBoxMatcher(
            thresholds=[bg_thresh_lo, bg_thresh_hi, fg_threshold],
            match_values=[-3, -2, -1, 1],
        )
        match_indices, matched_values = matcher(sim_matrix)
        positive_matches = tf.greater_equal(matched_values, 0)
        negative_matches = tf.equal(matched_values, -2)

        self.assertAllEqual(positive_matches.numpy(), [False, True])
        self.assertAllEqual(negative_matches.numpy(), [True, False])
        self.assertAllEqual(match_indices.numpy(), [0, 2])
        self.assertAllEqual(matched_values.numpy(), [-2, 1])

    def test_box_matcher_batched(self):
        sim_matrix = tf.constant([[[0.04, 0, 0, 0], [0, 0, 1.0, 0]]], dtype=tf.float32)

        fg_threshold = 0.5
        bg_thresh_hi = 0.2
        bg_thresh_lo = 0.0

        matcher = ArgmaxBoxMatcher(
            thresholds=[bg_thresh_lo, bg_thresh_hi, fg_threshold],
            match_values=[-3, -2, -1, 1],
        )
        match_indices, matched_values = matcher(sim_matrix)
        positive_matches = tf.greater_equal(matched_values, 0)
        negative_matches = tf.equal(matched_values, -2)

        self.assertAllEqual(positive_matches.numpy(), [[False, True]])
        self.assertAllEqual(negative_matches.numpy(), [[True, False]])
        self.assertAllEqual(match_indices.numpy(), [[0, 2]])
        self.assertAllEqual(matched_values.numpy(), [[-2, 1]])
