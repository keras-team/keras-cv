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

from keras_cv.metrics.f_scores import FBetaScore


class FBetaScoreTest(tf.test.TestCase):
    def test_config_fbeta(self):
        fbeta_obj = FBetaScore(num_classes=3, beta=0.5, threshold=0.3, average=None)
        self.assertEqual(fbeta_obj.beta, 0.5)
        self.assertEqual(fbeta_obj.average, None)
        self.assertEqual(fbeta_obj.threshold, 0.3)
        self.assertEqual(fbeta_obj.num_classes, 3)
        self.assertEqual(fbeta_obj.dtype, tf.float32)

        # Check save and restore config
        fbeta_obj2 = FBetaScore.from_config(fbeta_obj.get_config())
        self.assertEqual(fbeta_obj2.beta, 0.5)
        self.assertEqual(fbeta_obj2.average, None)
        self.assertEqual(fbeta_obj2.threshold, 0.3)
        self.assertEqual(fbeta_obj2.num_classes, 3)
        self.assertEqual(fbeta_obj2.dtype, tf.float32)

    def _test_tf(self, avg, beta, act, pred, sample_weights, threshold):
        act = tf.constant(act, tf.float32)
        pred = tf.constant(pred, tf.float32)

        fbeta = FBetaScore(3, avg, beta, threshold)
        fbeta.update_state(act, pred, sample_weights)
        return fbeta.result().numpy()

    def _test_fbeta_score(self, actuals, preds, sample_weights, avg, beta_val, result, threshold):
        tf_score = self._test_tf(avg, beta_val, actuals, preds, sample_weights, threshold)
        self.assertAllClose(tf_score, result)

    def test_fbeta_perfect_score(self):
        preds = [[0.7, 0.7, 0.7], [1, 0, 0], [0.9, 0.8, 0]]
        actuals = [[1, 1, 1], [1, 0, 0], [1, 1, 0]]

        for avg_val in ["micro", "macro", "weighted"]:
            for beta in [0.5, 1.0, 2.0]:
                self._test_fbeta_score(actuals, preds, None, avg_val, beta, 1.0, 0.66)

    def test_fbeta_worst_score(self):
        preds = [[0.7, 0.7, 0.7], [1, 0, 0], [0.9, 0.8, 0]]
        actuals = [[0, 0, 0], [0, 1, 0], [0, 0, 1]]

        for avg_val in ["micro", "macro", "weighted"]:
            for beta in [0.5, 1.0, 2.0]:
                self._test_fbeta_score(actuals, preds, None, avg_val, beta, 0.0, 0.66)

    def test_fbeta_random_score(self):
        preds = [[0.7, 0.7, 0.7], [1, 0, 0], [0.9, 0.8, 0]]
        actuals = [[0, 0, 1], [1, 1, 0], [1, 1, 1]]
        scenarios = [
            (None, 0.5, [0.71428573, 0.5, 0.833334]),
            (None, 1.0, [0.8, 0.5, 0.6666667]),
            (None, 2.0, [0.9090904, 0.5, 0.555556]),
            ("micro", 0.5, 0.6666667),
            ("micro", 1.0, 0.6666667),
            ("micro", 2.0, 0.6666667),
            ("macro", 0.5, 0.6825397),
            ("macro", 1.0, 0.6555555),
            ("macro", 2.0, 0.6548822),
            ("weighted", 0.5, 0.6825397),
            ("weighted", 1.0, 0.6555555),
            ("weighted", 2.0, 0.6548822),
        ]
        for avg_val, beta, result in scenarios:
            self._test_fbeta_score(actuals, preds, None, avg_val, beta, result, 0.66)

    def test_fbeta_random_score_none(self):
        preds = [
            [0.9, 0.1, 0],
            [0.2, 0.6, 0.2],
            [0, 0, 1],
            [0.4, 0.3, 0.3],
            [0, 0.9, 0.1],
            [0, 0, 1],
        ]
        actuals = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]]
        scenarios = [
            (None, 0.5, [0.9090904, 0.555556, 1.0]),
            (None, 1.0, [0.8, 0.6666667, 1.0]),
            (None, 2.0, [0.71428573, 0.833334, 1.0]),
            ("micro", 0.5, 0.833334),
            ("micro", 1.0, 0.833334),
            ("micro", 2.0, 0.833334),
            ("macro", 0.5, 0.821549),
            ("macro", 1.0, 0.822222),
            ("macro", 2.0, 0.849206),
            ("weighted", 0.5, 0.880471),
            ("weighted", 1.0, 0.844445),
            ("weighted", 2.0, 0.829365),
        ]
        for avg_val, beta, result in scenarios:
            self._test_fbeta_score(actuals, preds, None, avg_val, beta, result, None)

    def test_fbeta_weighted_random_score_none(self):
        preds = [
            [0.9, 0.1, 0],
            [0.2, 0.6, 0.2],
            [0, 0, 1],
            [0.4, 0.3, 0.3],
            [0, 0.9, 0.1],
            [0, 0, 1],
        ]
        actuals = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1]]
        scenarios = [
            (None, 0.5, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.909091, 0.555556, 1.0]),
            (None, 0.5, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0]),
            (None, 0.5, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], [0.9375, 0.714286, 1.0]),
            (None, 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.8, 0.666667, 1.0]),
            (None, 1.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0]),
            (None, 1.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], [0.857143, 0.8, 1.0]),
            (None, 2.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.714286, 0.833333, 1.0]),
            (None, 2.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0]),
            (None, 2.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], [0.789474, 0.909091, 1.0]),
            ("micro", 0.5, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.833333),
            ("micro", 0.5, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 1.0),
            ("micro", 0.5, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.9),
            ("micro", 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.833333),
            ("micro", 1.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 1.0),
            ("micro", 1.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.9),
            ("micro", 2.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.833333),
            ("micro", 2.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 1.0),
            ("micro", 2.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.9),
            ("macro", 0.5, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.821549),
            ("macro", 0.5, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 0.666667),
            ("macro", 0.5, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.883929),
            ("macro", 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.822222),
            ("macro", 1.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 0.666667),
            ("macro", 1.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.885714),
            ("macro", 2.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.849206),
            ("macro", 2.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 0.666667),
            ("macro", 2.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.899522),
            ("weighted", 0.5, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.880471),
            ("weighted", 0.5, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 1.0),
            ("weighted", 0.5, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.917857),
            ("weighted", 1.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.844444),
            ("weighted", 1.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 1.0),
            ("weighted", 1.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.902857),
            ("weighted", 2.0, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.829365),
            ("weighted", 2.0, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0], 1.0),
            ("weighted", 2.0, [0.5, 1.0, 1.0, 1.0, 0.5, 1.0], 0.897608),
        ]
        for avg_val, beta, sample_weights, result in scenarios:
            self._test_fbeta_score(actuals, preds, sample_weights, avg_val, beta, result, None)

    def test_keras_model(self):
        fbeta = FBetaScore(5, "micro", 1.0)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.keras.layers.Dense(5, activation="softmax"))
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["acc", fbeta]
        )
        data = tf.random.uniform((10, 3))
        labels = tf.random.uniform((10, 5))
        model.fit(data, labels, epochs=1, batch_size=5, verbose=0)