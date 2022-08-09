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

from keras_cv.losses import FocalLoss


class FocalTest(tf.test.TestCase):
    def test_output_shape(self):
        y_true = tf.cast(
            tf.random.uniform(shape=[2, 5], minval=0, maxval=2, dtype=tf.int32),
            tf.float32,
        )
        y_pred = tf.random.uniform(shape=[2, 5], minval=0, maxval=1, dtype=tf.float32)

        focal_loss = FocalLoss()

        self.assertAllEqual(focal_loss(y_true, y_pred).shape, [])

    def test_output_shape_reduction_none(self):
        y_true = tf.cast(
            tf.random.uniform(shape=[2, 5], minval=0, maxval=2, dtype=tf.int32),
            tf.float32,
        )
        y_pred = tf.random.uniform(shape=[2, 5], minval=0, maxval=1, dtype=tf.float32)

        focal_loss = FocalLoss(reduction="none")

        self.assertAllEqual(
            focal_loss(y_true, y_pred).shape,
            [
                2,
            ],
        )

    def test_output_shape_from_logits(self):
        y_true = tf.cast(
            tf.random.uniform(shape=[2, 5], minval=0, maxval=2, dtype=tf.int32),
            tf.float32,
        )
        y_pred = tf.random.uniform(
            shape=[2, 5], minval=-10, maxval=10, dtype=tf.float32
        )

        focal_loss = FocalLoss()

        self.assertAllEqual(focal_loss(y_true, y_pred).shape, [])

    def test_1d_output(self):
        y_true = [0.0, 1.0, 1.0]
        y_pred = [0.1, 0.7, 0.9]

        focal_loss = FocalLoss()

        self.assertAllClose(focal_loss(y_true, y_pred), 0.00302626)

    def test_1d_output_from_logits(self):
        y_true = [0.0, 1.0, 1.0]
        y_pred = [-2.1972246, 0.8472978, 2.1972241]

        focal_loss = FocalLoss(from_logits=True)

        self.assertAllClose(focal_loss(y_true, y_pred), 0.00302626)

    def test_from_logits_argument(self):
        y_true = tf.random.uniform((2, 8, 10))
        y_logits = tf.random.uniform((2, 8, 10), minval=-1000, maxval=1000)

        y_pred = tf.nn.sigmoid(y_logits)

        focal_loss_on_logits = FocalLoss(from_logits=True)
        focal_loss = FocalLoss()

        self.assertAllClose(
            focal_loss_on_logits(y_true, y_logits), focal_loss(y_true, y_pred)
        )
