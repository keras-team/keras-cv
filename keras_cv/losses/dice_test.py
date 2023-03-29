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


class DiceTest(tf.test.TestCase):
    def test_binary_output_shape(self):
        y_true = tf.random.uniform(
            shape=(3, 5, 5, 1), minval=0, maxval=1, dtype=tf.int32
        )
        y_pred = tf.random.uniform(
            shape=(3, 5, 5, 1), minval=0, maxval=2, dtype=tf.float32
        )

        binary_dice_loss = BinaryDice()
        self.assertAllEqual(binary_dice_loss(y_true, y_pred).shape, ())

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
