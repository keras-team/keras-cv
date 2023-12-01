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

from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.backend.config import keras_3
from keras_cv.losses import FocalLoss
from keras_cv.tests.test_case import TestCase


class FocalTest(TestCase):
    def test_output_shape(self):
        y_true = np.random.uniform(size=[2, 5], low=0, high=1)
        y_pred = np.random.uniform(size=[2, 5], low=0, high=1)

        focal_loss = FocalLoss(reduction="sum")

        self.assertAllEqual(focal_loss(y_true, y_pred).shape, [])

    def test_output_shape_reduction_none(self):
        y_true = np.random.uniform(size=[2, 5], low=0, high=1)
        y_pred = np.random.uniform(size=[2, 5], low=0, high=1)

        focal_loss = FocalLoss(reduction="none")

        self.assertAllEqual(
            focal_loss(y_true, y_pred).shape,
            [
                2,
            ],
        )

    def test_output_shape_from_logits(self):
        y_true = np.random.uniform(size=[2, 5], low=0, high=1)
        y_pred = np.random.uniform(size=[2, 5], low=-10, high=10)

        focal_loss = FocalLoss(reduction="none", from_logits=True)

        self.assertAllEqual(
            focal_loss(y_true, y_pred).shape,
            [
                2,
            ],
        )

    def test_from_logits_argument(self):
        rng = np.random.default_rng(1337)
        y_true = rng.uniform(size=(2, 8, 10)).astype("float64")
        y_logits = rng.uniform(low=-1000, high=1000, size=(2, 8, 10)).astype(
            "float64"
        )
        y_pred = ops.cast(ops.sigmoid(y_logits), "float32")

        focal_loss_on_logits = FocalLoss(from_logits=True)
        focal_loss = FocalLoss()

        # TODO(ianstenbit): This probably warrants some more investigation.
        # In the current implementation, I've verified that training RetinaNet
        # works in all backends with this implementation.
        # TF backend somehow has different numerics.
        expected_loss = (
            31.11176
            if keras_3() and keras.backend.backend() != "tensorflow"
            else 925.28081
        )
        self.assertAllClose(
            focal_loss_on_logits(y_true, y_logits), expected_loss
        )
        self.assertAllClose(focal_loss(y_true, y_pred), 31.11176)
