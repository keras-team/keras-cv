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

import numpy as np
from absl.testing import parameterized

import keras_cv
from keras_cv.src.backend import ops
from keras_cv.src.tests.test_case import TestCase


class CenterNetBoxLoss(TestCase):
    @parameterized.named_parameters(
        (
            "none",
            "none",
            (
                2,
                10,
            ),
        ),
        ("sum", "sum", ()),
        ("sum_over_batch_size", "sum_over_batch_size", ()),
    )
    def test_proper_output_shapes(self, reduction, target_size):
        loss = keras_cv.losses.CenterNetBoxLoss(
            num_heading_bins=4, anchor_size=[1.0, 1.0, 1.0], reduction=reduction
        )
        result = loss(
            y_true=np.random.uniform(size=(2, 10, 7)),
            # Predictions have xyz,lwh, and 2*4 values for heading.
            y_pred=np.random.uniform(size=(2, 10, 6 + 2 * 4)),
        )
        self.assertEqual(result.shape, target_size)

    def test_heading_regression_loss(self):
        num_heading_bins = 4
        loss = keras_cv.losses.CenterNetBoxLoss(
            num_heading_bins=num_heading_bins, anchor_size=[1.0, 1.0, 1.0]
        )
        heading_true = np.array(
            [[0, (1 / 2.0) * np.pi, np.pi, (3.0 / 2.0) * np.pi]]
        )
        heading_pred = np.array(
            [
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ]
        )
        heading_loss = loss.heading_regression_loss(
            heading_true=heading_true, heading_pred=heading_pred
        )
        ce_loss = -np.log(np.exp(1) / np.exp([1, 0, 0, 0]).sum())
        expected_loss = ce_loss * num_heading_bins
        self.assertAllClose(ops.sum(heading_loss), expected_loss)
