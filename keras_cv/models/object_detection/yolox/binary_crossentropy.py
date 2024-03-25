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


import warnings

import tensorflow as tf
from tensorflow import keras


class BinaryCrossentropy(keras.losses.Loss):
    """Computes the cross-entropy loss between true labels and predicted labels.

    Use this cross-entropy loss for binary (0 or 1) classification applications.
    This loss is updated for YoloX by offering support for no axis to mean over.

    Args:
        from_logits: Whether to interpret `y_pred` as a tensor of
            [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
            assume that `y_pred` contains probabilities (i.e., values in [0,
            1]).
        label_smoothing: Float in [0, 1]. When 0, no smoothing occurs. When >
            0, we compute the loss between the predicted labels and a smoothed
            version of the true labels, where the smoothing squeezes the labels
            towards 0.5.  Larger values of `label_smoothing` correspond to
            heavier smoothing.
        axis: the axis along which to mean the ious. Defaults to `no_reduction`
            which implies mean across no axes.

    Example:
    ```python
    model.compile(
      loss=BinaryCrossentropy(from_logits=True)
      ....
    )
    ```
    """

    def __init__(
        self, from_logits=False, label_smoothing=0.0, axis=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.axis = axis

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        label_smoothing = tf.convert_to_tensor(
            self.label_smoothing, dtype=y_pred.dtype
        )

        def _smooth_labels():
            return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

        y_true = tf.__internal__.smart_cond.smart_cond(
            label_smoothing, _smooth_labels, lambda: y_true
        )

        if self.axis == "no_reduction":
            warnings.warn(
                "`axis='no_reduction'` is a temporary API, and the API"
                "contract will be replaced in the future with a more generic "
                "solution covering all losses."
            )
            return tf.reduce_mean(
                keras.backend.binary_crossentropy(
                    y_true, y_pred, from_logits=self.from_logits
                ),
                axis=self.axis,
            )

        return keras.backend.binary_crossentropy(
            y_true, y_pred, from_logits=self.from_logits
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
                "axis": self.axis,
            }
        )
        return config
