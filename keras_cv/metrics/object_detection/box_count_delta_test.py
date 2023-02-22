# Copyright 2023 The KerasCV Authors
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

import keras_cv


class BoxCountDeltaTest(tf.test.TestCase):
    def test_dense_boxes(self):
        mean_box_count_delta = keras_cv.metrics.BoxCountDelta(
            mode="absolute", name="test-name"
        )
        y_true = {
            "classes": [[0, 0, -1]],
            "boxes": [[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]],
        }
        y_pred = {
            "classes": [[0, 1, -1, -1]],
            "boxes": [[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]],
        }
        mean_box_count_delta.update_state(y_true, y_pred)
        self.assertAllEqual(mean_box_count_delta.result(), 0.0)

        y_true = {
            "classes": [[0, -1, -1]],
            "boxes": [[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]],
        }
        y_pred = {
            "classes": [[0, 1, 1, -1]],
            "boxes": [[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]],
        }
        mean_box_count_delta.update_state(y_true, y_pred)
        self.assertAllEqual(mean_box_count_delta.result(), 1.0)

    def test_relative_mode(self):
        mean_box_count_delta = keras_cv.metrics.BoxCountDelta(
            mode="relative", name="test-name"
        )
        y_true = {
            "classes": [[0, 0, 0]],
            "boxes": [[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]],
        }
        y_pred = {
            "classes": [[0, -1, -1, -1]],
            "boxes": [[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]],
        }
        mean_box_count_delta.update_state(y_true, y_pred)
        self.assertAllEqual(mean_box_count_delta.result(), -2.0)

        y_true = {
            "classes": [[0, -1, -1]],
            "boxes": [[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]],
        }
        y_pred = {
            "classes": [[0, 1, 1, -1]],
            "boxes": [[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]],
        }
        mean_box_count_delta.update_state(y_true, y_pred)
        self.assertAllEqual(mean_box_count_delta.result(), 0.0)

        mean_box_count_delta.update_state(y_true, y_pred)
        self.assertAllEqual(
            mean_box_count_delta.result(), tf.cast(2 / 3, tf.float32)
        )

    def test_ragged_boxes(self):
        mean_box_count_delta = keras_cv.metrics.BoxCountDelta(
            mode="absolute", name="test-name"
        )
        y_true = {
            "classes": tf.ragged.stack(
                [
                    tf.constant([0, 0]),
                    tf.constant([0]),
                    tf.constant([0, 0, 0]),
                ]
            ),
            "boxes": tf.ragged.stack(
                [
                    tf.constant([[0, 0, 1, 1], [0, 0, 1, 1]]),
                    tf.constant([[0, 0, 1, 1]]),
                    tf.constant([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]),
                ]
            ),
        }
        y_pred = {
            "classes": tf.ragged.stack(
                [
                    tf.constant([0, 0, 0]),
                    tf.constant([0, 0]),
                    tf.constant([0, 0, 0, 0]),
                ]
            ),
            "boxes": tf.ragged.stack(
                [
                    tf.constant([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]),
                    tf.constant([[0, 0, 1, 1], [0, 0, 1, 1]]),
                    tf.constant(
                        [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]
                    ),
                ]
            ),
        }

        mean_box_count_delta.update_state(y_true, y_pred)
        self.assertAllEqual(mean_box_count_delta.result(), 1.0)
