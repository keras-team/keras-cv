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

from keras_cv.losses.giou_loss import GIoULoss


class GIoUTest(tf.test.TestCase):
    def test_output_shape(self):
        y_true = tf.random.uniform(
            shape=(2, 2, 4), minval=0, maxval=10, dtype=tf.int32
        )
        y_pred = tf.random.uniform(
            shape=(2, 2, 4), minval=0, maxval=20, dtype=tf.int32
        )

        giou_loss = GIoULoss(bounding_box_format="xywh")

        self.assertAllEqual(giou_loss(y_true, y_pred).shape, ())

    def test_output_shape_reduction_none(self):
        y_true = tf.random.uniform(
            shape=(2, 2, 4), minval=0, maxval=10, dtype=tf.int32
        )
        y_pred = tf.random.uniform(
            shape=(2, 2, 4), minval=0, maxval=20, dtype=tf.int32
        )

        giou_loss = GIoULoss(bounding_box_format="xywh", reduction="none")

        self.assertAllEqual(
            giou_loss(y_true, y_pred).shape,
            [
                2,
            ],
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

        giou_loss = GIoULoss(bounding_box_format="rel_xyxy")

        self.assertAllEqual(giou_loss(y_true, y_pred).shape, ())

    def test_output_value(self):
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

        iou_loss = GIoULoss(bounding_box_format="xywh")

        # expected value for these values is 0.6452381
        self.assertAllClose(iou_loss(y_true, y_pred), 0.6452381)
