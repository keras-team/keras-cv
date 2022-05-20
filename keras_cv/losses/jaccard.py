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


def gather_channels(*matrices, indices=None):
    # Gather channel axis according to the indices.
    if indices is None:
        return matrices

    gathered_channels = []

    for matrix in matrices:
        if keras.backend.image_data_format() == "channels_last":
            matrix = tf.gather(matrix, indices, axis=-1)
        else:
            matrix = tf.gather(matrix, indices, axis=1)
        gathered_channels.append(matrix)

    return gathered_channels


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class Jaccard(keras.losses.Loss):
    def __init__(
        self,
        from_logits=False,
        label_smoothing=0.0,
        class_ids=None,
        axis=[1, 2],
        epsilon=1e-07,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.class_ids = class_ids
        self.axis = axis
        self.epsilon = epsilon

    def _smooth_labels(self, y_true, y_pred, label_smoothing):
        num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
        return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, dtype=y_pred.dtype)
        label_smoothing = tf.convert_to_tensor(self.label_smoothing, dtype=y_pred.dtype)

        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)

        if tf.cast(label_smoothing, dtype=tf.bool):
            y_true = self._smooth_labels(y_true, y_pred, label_smoothing)

        if self.class_ids is not None:
            y_true, y_pred = gather_channels(y_true, y_pred, indices=self.class_ids)

        intersection = tf.reduce_sum(y_true * y_pred, axis=self.axis)
        union = tf.reduce_sum(y_true + y_pred, axis=self.axis) - intersection

        score = (intersection + self.epsilon) / (union + self.epsilon)
        return 1 - tf.reduce_mean(score)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "from_logits": self.from_logits,
                "class_ids": self.class_ids,
                "label_smoothing": self.label_smoothing,
                "epsilon": self.epsilon,
                "axis": self.axis,
            }
        )
        return config
