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
from tensorflow import keras

from keras_cv.losses.dice import CategoricalDice
from keras_cv.losses.dice import SparseDice
from keras_cv.losses.dice import BinaryDice

class DiceTest(tf.test.TestCase):
    def test_categorical_output_shape(self):

        projections_1 = tf.random.uniform(
            shape=(10, 128), minval=0, maxval=10, dtype=tf.float32
        )
        projections_2 = tf.random.uniform(
            shape=(10, 128), minval=0, maxval=10, dtype=tf.float32
        )
        categorical_dice_loss = CategoricalDice()
        self.assertAllEqual(categorical_dice_loss(projections_1, projections_2).shape, ())

    def test_sparse_output_shape(self):
        projections_1 = tf.random.uniform(
            shape=(10, 128), minval=0, maxval=10, dtype=tf.float32
        )
        projections_2 = tf.random.uniform(
            shape=(10, 128), minval=0, maxval=10, dtype=tf.float32
        )
        sparse_dice_loss = SparseDice()
        self.assertAllEqual(sparse_dice_loss(projections_1, projections_2).shape, ())

    def test_binary_output_shape(self):
        projections_1 = tf.random.uniform(
            shape=(10, 128), minval=0, maxval=10, dtype=tf.float32
        )
        projections_2 = tf.random.uniform(
            shape=(10, 128), minval=0, maxval=10, dtype=tf.float32
        )
        binary_dice_loss = BinaryDice()
        self.assertAllEqual(binary_dice_loss(projections_1, projections_2).shape, ())

    def test_binary_output_shape(self):
        projections_1 = tf.random.uniform(
            shape=(10, 128), minval=0, maxval=10, dtype=tf.float32
        )
        projections_2 = tf.random.uniform(
            shape=(10, 128), minval=0, maxval=10, dtype=tf.float32
        )
        binary_dice_loss = BinaryDice()
        self.assertAllEqual(binary_dice_loss(projections_1, projections_2).shape, ())

    def test_axis(self):
        self.assertAllEqual()

    def test_categorical_output_value(self):
        projections_1 = [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ]

        projections_2 = [
            [6.0, 5.0, 4.0, 3.0],
            [5.0, 4.0, 3.0, 2.0],
            [4.0, 3.0, 2.0, 1.0],
        ]

        categorical_loss = CategoricalDice()
        self.assertAllClose(categorical_loss(projections_1, projections_2), 3.566689)

        categorical_loss = CategoricalDice()
        self.assertAllClose(categorical_loss(projections_1, projections_2), 5.726100)

    def test_sparse_output_value(self):
        projections_1 = [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ]

        projections_2 = [
            [6.0, 5.0, 4.0, 3.0],
            [5.0, 4.0, 3.0, 2.0],
            [4.0, 3.0, 2.0, 1.0],
        ]

        sparse_loss = SparseDice()
        self.assertAllClose(sparse_loss(projections_1, projections_2), 3.566689)

        sparse_loss = SparseDice()
        self.assertAllClose(sparse_loss(projections_1, projections_2), 5.726100)

    def test_binary_output_value(self):
        projections_1 = [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ]

        projections_2 = [
            [6.0, 5.0, 4.0, 3.0],
            [5.0, 4.0, 3.0, 2.0],
            [4.0, 3.0, 2.0, 1.0],
        ]

        binary_loss = BinaryDice()
        self.assertAllClose(binary_loss(projections_1, projections_2), 3.566689)

        binary_loss = BinaryDice()
        self.assertAllClose(binary_loss(projections_1, projections_2), 5.726100)
