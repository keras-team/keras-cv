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

from keras_cv.metrics.f_scores import SparseF1Score, SparseFBetaScore


class SparseF1ScoreTest(tf.test.TestCase):
    def test_eq(self):
        f1 = SparseF1Score(3)
        fbeta = SparseFBetaScore(3, beta=1.0)

        preds = [
            [0.9, 0.1, 0],
            [0.2, 0.6, 0.2],
            [0, 0, 1],
            [0.4, 0.3, 0.3],
            [0, 0.9, 0.1],
            [0, 0, 1],
        ]
        actuals = [0, 1, 2, 0, 1, 2]

        fbeta.update_state(actuals, preds)
        f1.update_state(actuals, preds)
        self.assertAllEqual(fbeta.result().numpy(), f1.result().numpy())

    def test_sample_eq(self):
        f1 = SparseF1Score(3)
        f1_weighted = SparseF1Score(3)

        preds = [
            [0.9, 0.1, 0],
            [0.2, 0.6, 0.2],
            [0, 0, 1],
            [0.4, 0.3, 0.3],
            [0, 0.9, 0.1],
            [0, 0, 1],
        ]
        actuals = [0, 1, 2, 0, 0, 2]
        sample_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        f1.update_state(actuals, preds)
        f1_weighted(actuals, preds, sample_weights)
        self.assertAllEqual(f1.result().numpy(), f1_weighted.result().numpy())

    def test_config_f1(self):
        f1 = SparseF1Score(3)
        config = f1.get_config()
        self.assertNotIn("beta", config)