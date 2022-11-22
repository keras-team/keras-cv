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

from keras_cv.losses.dice import BinaryDice
from keras_cv.losses.dice import CategoricalDice
from keras_cv.losses.dice import SparseDice


class DiceTest(tf.test.TestCase):
    def test_categorical_output_shape(self):
        """
        Both y_true and y_pred are one-hot vectors
        """

        y_true = tf.random.uniform(
            shape=(3, 10, 10, 5), minval=0, maxval=5, dtype=tf.int32
        )

        y_pred = tf.random.uniform(
            shape=(3, 10, 10, 5), minval=0, maxval=5, dtype=tf.float32
        )

        categorical_dice_loss = CategoricalDice()
        self.assertAllEqual(categorical_dice_loss(y_true, y_pred).shape, ())

        sample_weight = [[0.5, 0.5, 0.2, 0.2, 0.5]]
        self.assertAllEqual(
            categorical_dice_loss(y_true, y_pred, sample_weight=sample_weight).shape, ()
        )

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

    def test_binary_output_shape(self):
        y_true = tf.random.uniform(
            shape=(3, 5, 5, 1), minval=0, maxval=1, dtype=tf.int32
        )
        y_pred = tf.random.uniform(
            shape=(3, 5, 5, 1), minval=0, maxval=2, dtype=tf.float32
        )

        binary_dice_loss = BinaryDice()
        self.assertAllEqual(binary_dice_loss(y_true, y_pred).shape, ())

    def test_categorical_output_value(self):
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

        y_true = tf.one_hot(y_true, depth=3)

        categorical_dice_loss = CategoricalDice(from_logits=True)
        self.assertAllClose(categorical_dice_loss(y_true, y_pred), 0.8834148)

        categorical_dice_loss = CategoricalDice(from_logits=False)
        self.assertAllClose(categorical_dice_loss(y_true, y_pred), 0.6666666)

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

    def test_binary_output_value(self):
        # All wrong
        y_true = np.array(
            [[[[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]]]]
        )
        y_pred = np.array(
            [
                [
                    [[1.0], [1.0], [1.0]],
                    [[1.0], [1.0], [1.0]],
                    [[1.0], [1.0], [1.0]],
                ]
            ]
        )

        binary_dice_loss = BinaryDice(from_logits=False)
        self.assertAllClose(binary_dice_loss(y_true, y_pred), 0.999999988888889)
        binary_dice_loss = BinaryDice(from_logits=True)
        self.assertAllClose(binary_dice_loss(y_true, y_pred), 0.9999999848013398)

        # All right
        y_true = np.array(
            [[[[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]]]]
        )
        y_pred = np.array(
            [
                [
                    [[0.0], [0.0], [0.0]],
                    [[0.0], [0.0], [0.0]],
                    [[0.0], [0.0], [0.0]],
                ]
            ]
        )

        binary_dice_loss = BinaryDice(from_logits=False)
        self.assertAllClose(binary_dice_loss(y_true, y_pred), 0.0)
        binary_dice_loss = BinaryDice(from_logits=True)
        self.assertAllClose(binary_dice_loss(y_true, y_pred), 0.9999999777777783)
