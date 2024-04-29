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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops


@keras_cv_export("keras_cv.losses.SmoothL1Loss")
class SmoothL1Loss(keras.losses.Loss):
    """Implements Smooth L1 loss.

    SmoothL1Loss implements the SmoothL1 function, where values less than
    `l1_cutoff` contribute to the overall loss based on their squared
    difference, and values greater than l1_cutoff contribute based on their raw
    difference.

    Args:
        l1_cutoff: differences between y_true and y_pred that are larger than
            `l1_cutoff` are treated as `L1` values
    """

    def __init__(self, l1_cutoff=1.0, **kwargs):
        super().__init__(**kwargs)
        self.l1_cutoff = l1_cutoff

    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = ops.abs(difference)
        squared_difference = difference**2
        loss = ops.where(
            absolute_difference < self.l1_cutoff,
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return ops.mean(loss, axis=-1)

    def get_config(self):
        config = {
            "l1_cutoff": self.l1_cutoff,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
