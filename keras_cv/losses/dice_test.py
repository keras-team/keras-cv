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
import tensorflow as tf

from keras_cv.losses.dice import SparseDice


class DiceTest(tf.test.TestCase):
    def test_sparse_output_shape(self):
        """
        y_true is a sparse vector, y_pred is a one-hot vector
        """
        y_true = tf.random.uniform(
            shape=(3, 10, 10, 1), minval=0, maxval=5, dtype=tf.int32
        )

        y_pred = tf.random.uniform(
            shape=(3, 10, 10, 5), minval=0, maxval=5, dtype=tf.float32
        )

        sparse_dice_loss = SparseDice()
        self.assertAllEqual(sparse_dice_loss(y_true, y_pred).shape, ())

        sample_weight = [[0.5, 0.5, 0.2, 0.2, 0.5]]
        self.assertAllEqual(
            sparse_dice_loss(y_true, y_pred, sample_weight=sample_weight).shape, ()
        )

    def test_sparse_output_value(self):
        y_true = np.array([[[[0], [0], [0]], [[0], [0], [0]], [[0], [0], [0]]]])

        y_pred = np.array(
            [
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ]
            ]
        )

        sparse_dice_loss = SparseDice(from_logits=False)
        self.assertAllClose(sparse_dice_loss(y_true, y_pred), 0.3333333296296296)
        sparse_dice_loss = SparseDice(from_logits=True)
        self.assertAllClose(sparse_dice_loss(y_true, y_pred), 0.833333309722223)

        y_true = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

        y_pred = np.array(
            [
                [
                    [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                    [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                    [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                ]
            ]
        )

        sparse_dice_loss = SparseDice(from_logits=False)
        self.assertAllClose(sparse_dice_loss(y_true, y_pred), 0.6666666592592594)
        sparse_dice_loss = SparseDice(from_logits=True)
        self.assertAllClose(sparse_dice_loss(y_true, y_pred), 0.8834148377575873)

        y_true = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]])

        y_pred = np.array(
            [
                [
                    [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                    [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                    [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                ]
            ]
        )

        sparse_dice_loss = SparseDice(from_logits=False)
        self.assertAllClose(sparse_dice_loss(y_true, y_pred), 0.0)
        sparse_dice_loss = SparseDice(from_logits=True)
        self.assertAllClose(sparse_dice_loss(y_true, y_pred), 0.7563137715411146)
