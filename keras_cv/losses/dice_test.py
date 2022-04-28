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

from keras_cv.losses import Dice


class DiceTest(tf.test.TestCase):
    def test_on_2D_model():
        def get_2d_model(num_classes, activation=None):
            input = keras.Input(shape=(None, None, 3))
            output = keras.layers.Conv2D(num_classes, 1, activation=activation)(input)
            model = keras.Model(input, output)
            return model

        # multi-class classification with activation to None.
        num_classes = 4
        activation = None
        model = get_2d_model(num_classes=num_classes, activation=activation)
        model.compile(loss=Dice(from_logits=True))
        model.fit(
            x=tf.random.uniform(shape=(2, 5, 5, 3)),
            y=tf.one_hot(
                tf.random.uniform(shape=[2, 5, 5], minval=0, maxval=5, dtype=tf.int32),
                depth=num_classes,
            ),
            verbose=0,
        )

        # multi-class classification with activation to softmax.
        num_classes = 10
        activation = "softmax"
        model = get_2d_model(num_classes=num_classes, activation=activation)
        model.compile(loss=Dice(from_logits=False))
        model.fit(
            x=tf.random.uniform(shape=(2, 5, 5, 3)),
            y=tf.one_hot(
                tf.random.uniform(shape=[2, 5, 5], minval=0, maxval=5, dtype=tf.int32),
                depth=num_classes,
            ),
            verbose=0,
        )

        # binary classification with activation to sigmoid.
        num_classes = 1
        activation = "sigmoid"
        model = get_2d_model(num_classes=num_classes, activation=activation)
        model.compile(loss=Dice(from_logits=False))
        model.fit(
            x=tf.random.uniform(shape=(2, 5, 5, 3)),
            y=tf.random.uniform(
                shape=[2, 5, 5, num_classes], minval=0, maxval=2, dtype=tf.int32
            ),
            verbose=0,
        )

    def test_on_3D_model():
        def get_3d_model(num_classes, activation=None):
            input = keras.Input(shape=(None, None, None, 3))
            output = keras.layers.Conv3D(num_classes, 1, activation=activation)(input)
            model = keras.Model(input, output)
            return model

        num_classes = 10
        activation = None
        model = get_3d_model(num_classes=num_classes, activation=activation)
        model.compile(loss=Dice(from_logits=True))

        model.fit(
            x=tf.random.uniform(shape=(2, 5, 5, 4, 3)),
            y=tf.one_hot(
                tf.random.uniform(
                    shape=[2, 5, 5, 4], minval=0, maxval=5, dtype=tf.int32
                ),
                depth=num_classes,
            ),
            verbose=0,
        )
