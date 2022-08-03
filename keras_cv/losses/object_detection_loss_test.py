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

import pytest
import tensorflow as tf
from absl.testing import parameterized

import keras_cv


class ObjectDetectionLossTest(tf.test.TestCase, parameterized.TestCase):
    def test_requires_proper_bounding_box_shapes(self):
        loss=keras_cv.losses.ObjectDetectionLoss(
            classes=20,
            classification_loss=keras_cv.losses.FocalLoss(from_logits=True, reduction='none'),
            box_loss=keras_cv.losses.SmoothL1Loss(cutoff=1.0, reduction='none'),
            reduction="auto"
        )

        with self.assertRaisesRegex(ValueError, "y_true should have shape"):
            loss(
                y_true=tf.random.uniform((20, 300, 24)),
                y_pred=tf.random.uniform((20, 300, 24)),
            )

        with self.assertRaisesRegex(ValueError, "y_pred should have shape"):
            loss(
                y_true=tf.random.uniform((20, 300, 5)),
                y_pred=tf.random.uniform((20, 300, 6)),
            )

    @parameterized.named_parameters(
        ("none", "none", (20,)),
        ("sum", "sum", ()),
        ("sum_over_batch_size", "sum_over_batch_size", ()),
    )
    def test_proper_output_shapes(self, reduction, target_size):
        loss=keras_cv.losses.ObjectDetectionLoss(
            classes=20,
            classification_loss=keras_cv.losses.FocalLoss(from_logits=True, reduction='none'),
            box_loss=keras_cv.losses.SmoothL1Loss(cutoff=1.0, reduction='none'),
            reduction=reduction
        )
        result = loss(
            y_true=tf.random.uniform((20, 300, 5)),
            y_pred=tf.random.uniform((20, 300, 24)),
        )
        self.assertEqual(result.shape, target_size)
