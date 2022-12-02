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


# --- Implementing Smooth L1 loss and Focal Loss as keras custom losses ---
class SmoothL1Loss(tf.keras.losses.Loss):
    """Implements Smooth L1 loss.

    SmoothL1Loss implements the SmoothL1 function, where values less than `beta`
    contribute to the overall loss based on their squared difference, and values greater
    than `beta` contribute based on their raw difference.

    Args:
        beta: differences between y_true and y_pred that are larger than `beta` are
            treated as `L1` values
    """

    def __init__(self, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference**2
        loss = tf.where(
            absolute_difference < self.beta,
            (0.5 * squared_difference) / self.beta,
            (absolute_difference - 0.5) * self.beta,
        )
        return tf.keras.backend.mean(loss, axis=-1)

    def get_config(self):
        config = {
            "beta": self.beta,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
