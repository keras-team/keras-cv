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

from keras_cv.losses.dice import Dice


def get_2d_model(num_classes, activation=None):
    input = keras.Input(shape=(None, None, 3))
    output = keras.layers.Conv2D(num_classes, 1, activation=activation)(input)
    model = keras.Model(input, output)
    return model


class DiceTest(tf.test.TestCase):
    def test_dice_score(self):
        y_true = tf.constant(
            [
                [
                    [
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0],
                    ],
                    [
                        [1.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0],
                    ],
                ],
                [
                    [
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0],
                    ],
                    [
                        [1.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0],
                    ],
                ],
            ]
        )

        y_pred = tf.constant(
            [
                [
                    [
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0],
                    ],
                    [
                        [1.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0],
                    ],
                ],
                [
                    [
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0],
                    ],
                    [
                        [1.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                ],
            ]
        )

        dice = Dice(from_logits=False, per_sample=True)
        score = dice(y_true, y_pred)
        self.assertAlmostEqual(score.numpy(), 0.22222221)

        dice = Dice(from_logits=False, per_sample=False)
        score = dice(y_true, y_pred)
        self.assertAlmostEqual(score.numpy(), 0.3888889)

    def test_output_shape(self):
        num_classes = 4
        activation = None

        model = get_2d_model(num_classes=num_classes, activation=activation)
        model.compile(loss=Dice(from_logits=True))

        y_true = tf.one_hot(
            tf.random.uniform(shape=[2, 5, 5], minval=0, maxval=5, dtype=tf.int32),
            depth=num_classes,
        ).shape

        y_pred = model(tf.random.uniform(shape=(2, 5, 5, 3)), training=False).shape

        self.assertAllEqual(y_true, y_pred)
