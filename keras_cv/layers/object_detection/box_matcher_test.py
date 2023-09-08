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

import numpy as np

from keras_cv.backend import ops
from keras_cv.layers.object_detection.box_matcher import BoxMatcher
from keras_cv.tests.test_case import TestCase


class BoxMatcherTest(TestCase):
    def test_box_matcher_invalid_length(self):
        fg_threshold = 0.5
        bg_thresh_hi = 0.2
        bg_thresh_lo = 0.0

        with self.assertRaisesRegex(ValueError, "must be len"):
            _ = BoxMatcher(
                thresholds=[bg_thresh_lo, bg_thresh_hi, fg_threshold],
                match_values=[-3, -2, -1],
            )

    def test_box_matcher_unsorted_thresholds(self):
        fg_threshold = 0.5
        bg_thresh_hi = 0.2
        bg_thresh_lo = 0.0

        with self.assertRaisesRegex(ValueError, "must be sorted"):
            _ = BoxMatcher(
                thresholds=[bg_thresh_hi, bg_thresh_lo, fg_threshold],
                match_values=[-3, -2, -1, 1],
            )

    def test_box_matcher_unbatched(self):
        sim_matrix = np.array([[0.04, 0, 0, 0], [0, 0, 1.0, 0]])

        fg_threshold = 0.5
        bg_thresh_hi = 0.2
        bg_thresh_lo = 0.0

        matcher = BoxMatcher(
            thresholds=[bg_thresh_lo, bg_thresh_hi, fg_threshold],
            match_values=[-3, -2, -1, 1],
        )
        match_indices, matched_values = matcher(sim_matrix)
        positive_matches = ops.greater_equal(matched_values, 0)
        negative_matches = ops.equal(matched_values, -2)

        self.assertAllEqual(positive_matches, [False, True])
        self.assertAllEqual(negative_matches, [True, False])
        self.assertAllEqual(match_indices, [0, 2])
        self.assertAllEqual(matched_values, [-2, 1])

    def test_box_matcher_batched(self):
        sim_matrix = np.array([[[0.04, 0, 0, 0], [0, 0, 1.0, 0]]])

        fg_threshold = 0.5
        bg_thresh_hi = 0.2
        bg_thresh_lo = 0.0

        matcher = BoxMatcher(
            thresholds=[bg_thresh_lo, bg_thresh_hi, fg_threshold],
            match_values=[-3, -2, -1, 1],
        )
        match_indices, matched_values = matcher(sim_matrix)
        positive_matches = ops.greater_equal(matched_values, 0)
        negative_matches = ops.equal(matched_values, -2)

        self.assertAllEqual(positive_matches, [[False, True]])
        self.assertAllEqual(negative_matches, [[True, False]])
        self.assertAllEqual(match_indices, [[0, 2]])
        self.assertAllEqual(matched_values, [[-2, 1]])

    def test_box_matcher_force_match(self):
        sim_matrix = np.array(
            [[0, 0.04, 0, 0.1], [0, 0, 1.0, 0], [0.1, 0, 0, 0], [0, 0, 0, 0.6]],
        )

        fg_threshold = 0.5
        bg_thresh_hi = 0.2
        bg_thresh_lo = 0.0

        matcher = BoxMatcher(
            thresholds=[bg_thresh_lo, bg_thresh_hi, fg_threshold],
            match_values=[-3, -2, -1, 1],
            force_match_for_each_col=True,
        )
        match_indices, matched_values = matcher(sim_matrix)
        positive_matches = ops.greater_equal(matched_values, 0)
        negative_matches = ops.equal(matched_values, -2)

        self.assertAllEqual(positive_matches, [True, True, True, True])
        self.assertAllEqual(negative_matches, [False, False, False, False])
        # the first anchor cannot be matched to 4th gt box given that is matched
        # to the last anchor.
        self.assertAllEqual(match_indices, [1, 2, 0, 3])
        self.assertAllEqual(matched_values, [1, 1, 1, 1])

    def test_box_matcher_empty_gt_boxes(self):
        sim_matrix = np.array([[], []])

        fg_threshold = 0.5
        bg_thresh_hi = 0.2
        bg_thresh_lo = 0.0

        matcher = BoxMatcher(
            thresholds=[bg_thresh_lo, bg_thresh_hi, fg_threshold],
            match_values=[-3, -2, -1, 1],
        )
        match_indices, matched_values = matcher(sim_matrix)
        positive_matches = ops.greater_equal(matched_values, 0)
        ignore_matches = ops.equal(matched_values, -1)

        self.assertAllEqual(positive_matches, [False, False])
        self.assertAllEqual(ignore_matches, [True, True])
        self.assertAllEqual(match_indices, [0, 0])
        self.assertAllEqual(matched_values, [-1, -1])
