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
"""Tests for COCOMeanAveragePrecision."""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras_cv.metrics import COCOMeanAveragePrecision


class COCOMeanAveragePrecisionTest(tf.test.TestCase):
    def test_runs_inside_model(self):
        i = keras.layers.Input((None, None, 6))
        model = keras.Model(i, i)

        mean_average_precision = COCOMeanAveragePrecision(
            max_detections=100,
            class_ids=[1],
            area_range=(0, 64**2),
        )

        # These would match if they were in the area range
        y_true = np.array([[[0, 0, 10, 10, 1], [5, 5, 10, 10, 1]]]).astype(np.float32)
        y_pred = np.array([[[0, 0, 10, 10, 1, 1.0], [5, 5, 10, 10, 1, 0.9]]]).astype(
            np.float32
        )

        model.compile(metrics=[mean_average_precision])
        model.evaluate(y_pred, y_true)

        self.assertAllEqual(mean_average_precision.result(), 1.0)
