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

from keras_cv.losses import BinaryPenaltyReducedFocalCrossEntropy
from keras_cv.tests.test_case import TestCase


class BinaryPenaltyReducedFocalLossTest(TestCase):
    def test_output_shape(self):
        y_true = (np.random.uniform(size=[2, 5], low=0, high=2),)
        y_pred = np.random.uniform(size=[2, 5], low=0, high=1)

        focal_loss = BinaryPenaltyReducedFocalCrossEntropy(reduction="sum")

        self.assertAllEqual(focal_loss(y_true, y_pred).shape, [])

    def test_output_shape_reduction_none(self):
        y_true = np.random.uniform(size=[2, 5], low=0, high=2)
        y_pred = np.random.uniform(size=[2, 5], low=0, high=2)

        focal_loss = BinaryPenaltyReducedFocalCrossEntropy(reduction="none")

        self.assertAllEqual(
            [2, 5],
            focal_loss(y_true, y_pred).shape,
        )

    def test_output_with_pos_label_pred(self):
        y_true = np.array([1.0])
        y_pred = np.array([1.0])
        focal_loss = BinaryPenaltyReducedFocalCrossEntropy(reduction="sum")
        self.assertAllClose(0.0, focal_loss(y_true, y_pred))

    def test_output_with_pos_label_neg_pred(self):
        y_true = np.array([1.0])
        y_pred = np.array([np.exp(-1.0)])
        focal_loss = BinaryPenaltyReducedFocalCrossEntropy(reduction="sum")
        # (1-1/e)^2 * log(1/e)
        self.assertAllClose(
            np.square(1 - np.exp(-1.0)), focal_loss(y_true, y_pred)
        )

    def test_output_with_neg_label_pred(self):
        y_true = np.array([0.0])
        y_pred = np.array([0.0])
        focal_loss = BinaryPenaltyReducedFocalCrossEntropy(reduction="sum")
        self.assertAllClose(0.0, focal_loss(y_true, y_pred))

    def test_output_with_neg_label_pos_pred(self):
        y_true = np.array([0.0])
        y_pred = np.array([1.0 - np.exp(-1.0)])
        focal_loss = BinaryPenaltyReducedFocalCrossEntropy(reduction="sum")
        # (1-0)^4 * (1-1/e)^2 * log(1/e)
        self.assertAllClose(
            np.square(1 - np.exp(-1.0)), focal_loss(y_true, y_pred)
        )

    def test_output_with_weak_label_pos_pred(self):
        y_true = np.array([0.5])
        y_pred = np.array([1.0 - np.exp(-1.0)])
        focal_loss = BinaryPenaltyReducedFocalCrossEntropy(
            beta=2.0, reduction="sum"
        )
        # (1-0.5)^2 * (1-1/e)^2 * log(1/e)
        self.assertAllClose(
            0.25 * np.square(1 - np.exp(-1.0)), focal_loss(y_true, y_pred)
        )

    def test_output_with_sample_weight(self):
        y_true = np.array([0.0])
        y_pred = np.array([1.0 - np.exp(-1.0)])
        sample_weight = np.array([0.5])
        focal_loss = BinaryPenaltyReducedFocalCrossEntropy(reduction="sum")
        # (1-0)^4 * (1-1/e)^2 * log(1/e)
        self.assertAllClose(
            0.5 * np.square(1 - np.exp(-1.0)),
            focal_loss(y_true, y_pred, sample_weight=sample_weight),
        )
