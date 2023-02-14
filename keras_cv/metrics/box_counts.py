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
import tensorflow.keras as keras
import tensorflow.keras.initializers as initializers

from keras_cv import bounding_box


class BoxCounts(keras.metrics.Metric):
    """BoxCounts computes average predictions per image."""

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ground_truths = self.add_weight(
            name="ground_truths",
            shape=(),
            dtype=tf.float32,
            initializer=initializers.Zeros(),
        )
        self.predictions = self.add_weight(
            name="predictions",
            shape=(),
            dtype=tf.float32,
            initializer=initializers.Zeros(),
        )
        self.samples = self.add_weight(
            name="samples",
            shape=(),
            dtype=tf.float32,
            initializer=initializers.Zeros(),
        )

    def reset_state(self):
        self.ground_truths.assign(tf.zeros_like(self.ground_truths))
        self.predictions.assign(tf.zeros_like(self.predictions))
        self.samples.assign(tf.zeros_like(self.samples))

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Args:
            y_true: a bounding box Tensor in corners format.
            y_pred: a bounding box Tensor in corners format.
            sample_weight: Currently unsupported.
        """
        del sample_weight
        y_pred = bounding_box.to_ragged(y_pred)
        y_true = bounding_box.to_ragged(y_true)
        for image in tf.range(tf.shape(y_pred["classes"])[0]):
            self.predictions.assign_add(
                tf.cast(tf.shape(y_pred["classes"][image])[0], tf.float32)
            )
            self.ground_truths.assign_add(
                tf.cast(tf.shape(y_true["classes"][image])[0], tf.float32)
            )

        self.samples.assign_add(tf.cast(tf.shape(y_pred["classes"])[0], tf.float32))

    def result(self):
        return {
            "predicted_boxes": self.predictions / self.samples,
            "true_boxes": self.ground_truths / self.samples,
        }

    def get_config(self):
        return super().get_config()
