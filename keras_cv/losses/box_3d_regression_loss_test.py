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

import tensorflow as tf
from absl.testing import parameterized

import keras_cv


class Box3DRegressionLossTest(tf.test.TestCase, parameterized.TestCase):
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
        loss = keras_cv.losses.Box3DRegressionLoss(
            num_heading_bins=4, anchor_size=[1.0, 1.0, 1.0], reduction=reduction
        )
        result = loss(
            y_true=tf.random.uniform((2, 10, 7)),
            # Predictions have xyz,lwh, and 2*4 values for heading.
            y_pred=tf.random.uniform((2, 10, 6 + 2 * 4)),
        )
        self.assertEqual(result.shape, target_size)
