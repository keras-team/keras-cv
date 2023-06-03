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
import tensorflow as tf
from absl.testing import parameterized

from keras_cv.losses.ciou_loss import CIoULoss


class CIoUTest(tf.test.TestCase, parameterized.TestCase):
    def test_output_shape(self):
        y_true = tf.random.uniform(
            shape=(2, 2, 4), minval=0, maxval=10, dtype=tf.int32
        )
        y_pred = tf.random.uniform(
            shape=(2, 2, 4), minval=0, maxval=20, dtype=tf.int32
        )

        ciou_loss = CIoULoss(bounding_box_format="xywh")

        self.assertAllEqual(ciou_loss(y_true, y_pred).shape, ())

    def test_output_shape_reduction_none(self):
        y_true = tf.random.uniform(
            shape=(2, 2, 4), minval=0, maxval=10, dtype=tf.int32
        )
        y_pred = tf.random.uniform(
            shape=(2, 2, 4), minval=0, maxval=20, dtype=tf.int32
        )

        ciou_loss = CIoULoss(bounding_box_format="xyxy", reduction="none")

        self.assertAllEqual(
            [2, 2],
            ciou_loss(y_true, y_pred).shape,
        )

    def test_output_shape_relative_formats(self):
        y_true = [
            [0.0, 0.0, 0.1, 0.1],
            [0.0, 0.0, 0.2, 0.3],
            [0.4, 0.5, 0.5, 0.6],
            [0.2, 0.2, 0.3, 0.3],
        ]

        y_pred = [
            [0.0, 0.0, 0.5, 0.6],
            [0.0, 0.0, 0.7, 0.3],
            [0.4, 0.5, 0.5, 0.6],
            [0.2, 0.1, 0.3, 0.3],
        ]

        ciou_loss = CIoULoss(bounding_box_format="rel_xyxy")

        self.assertAllEqual(ciou_loss(y_true, y_pred).shape, ())

    @parameterized.named_parameters(
        ("xyxy", "xyxy"),
        ("rel_xyxy", "rel_xyxy"),
    )
    def test_output_value(self, name):
        y_true = [
            [0, 0, 1, 1],
            [0, 0, 2, 3],
            [4, 5, 3, 6],
            [2, 2, 3, 3],
        ]

        y_pred = [
            [0, 0, 5, 6],
            [0, 0, 7, 3],
            [4, 5, 5, 6],
            [2, 1, 3, 3],
        ]
        expected_loss = 1.03202
        ciou_loss = CIoULoss(bounding_box_format="xyxy")
        if name == "rel_xyxy":
            scale_factor = 1 / 640.0
            y_true_scaled = np.array(y_true) * scale_factor
            y_pred_scaled = np.array(y_pred) * scale_factor

            y_true = tf.constant(y_true_scaled, dtype=tf.float32)
            y_pred = tf.constant(y_pred_scaled, dtype=tf.float32)

        self.assertAllClose(
            ciou_loss(y_true, y_pred), expected_loss, atol=0.0001
        )
