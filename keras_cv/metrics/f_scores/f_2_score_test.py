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
"""Tests for F beta scores."""

import tensorflow as tf

from keras_cv.metrics.f_scores import F2Score, FBetaScore


class F2ScoreTest(tf.test.TestCase):
    def test_eq(self):
        f2 = F2Score(3)
        fbeta = FBetaScore(3, beta=2.0)

        preds = [
            [0.9, 0.1, 0],
            [0.2, 0.6, 0.2],
            [0, 0, 1],
            [0.4, 0.3, 0.3],
            [0, 0.9, 0.1],
            [0, 0, 1],
        ]
        actuals = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]]

        fbeta.update_state(actuals, preds)
        f2.update_state(actuals, preds)
        self.assertAllEqual(fbeta.result().numpy(), f2.result().numpy())

    def test_sample_eq(self):
        f2 = F2Score(3)
        f2_weighted = F2Score(3)

        preds = [
            [0.9, 0.1, 0],
            [0.2, 0.6, 0.2],
            [0, 0, 1],
            [0.4, 0.3, 0.3],
            [0, 0.9, 0.1],
            [0, 0, 1],
        ]
        actuals = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]]
        sample_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        f2.update_state(actuals, preds)
        f2_weighted(actuals, preds, sample_weights)
        self.assertAllEqual(f2.result().numpy(), f2_weighted.result().numpy())

    def test_keras_model_f2(self):
        f2 = F2Score(5)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.keras.layers.Dense(5, activation="softmax"))
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["acc", f2]
        )
        data = tf.random.uniform((10, 3))
        labels = tf.random.uniform((10, 5))
        model.fit(data, labels, epochs=1, batch_size=5, verbose=0)

    def test_config_f2(self):
        f2 = F2Score(3)
        config = f2.get_config()
        self.assertNotIn("beta", config)