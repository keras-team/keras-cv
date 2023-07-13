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

from keras_cv.losses.simclr_loss import SimCLRLoss
from keras_cv.tests.test_case import TestCase


class SimCLRLossTest(TestCase):
    def test_output_shape(self):
        projections_1 = np.random.uniform(size=(10, 128), low=0, high=10)
        projections_2 = np.random.uniform(size=(10, 128), low=0, high=10)

        simclr_loss = SimCLRLoss(temperature=1)

        self.assertAllEqual(simclr_loss(projections_1, projections_2).shape, ())

    def test_output_shape_reduction_none(self):
        projections_1 = np.random.uniform(size=(10, 128), low=0, high=10)
        projections_2 = np.random.uniform(size=(10, 128), low=0, high=10)

        simclr_loss = SimCLRLoss(temperature=1, reduction="none")

        self.assertAllEqual(
            simclr_loss(projections_1, projections_2).shape, (10,)
        )

    def test_output_value(self):
        projections_1 = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
            ]
        )

        projections_2 = np.array(
            [
                [6.0, 5.0, 4.0, 3.0],
                [5.0, 4.0, 3.0, 2.0],
                [4.0, 3.0, 2.0, 1.0],
            ]
        )

        simclr_loss = SimCLRLoss(temperature=0.5)
        self.assertAllClose(simclr_loss(projections_1, projections_2), 3.566689)

        simclr_loss = SimCLRLoss(temperature=0.1)
        self.assertAllClose(simclr_loss(projections_1, projections_2), 5.726100)
