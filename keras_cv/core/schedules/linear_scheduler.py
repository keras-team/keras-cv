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


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class LinearScheduler:
    """LinearScheduler returns the value of variable based on the linear schedule.
    Args:
        initial_val: float, initial value of variable
        end_val: float, final value of variable
        total_steps: int, total number of steps
    Usage:
    ```python
    linear_scheduler = keras_cv.core.LinearScheduler(0.05, 0.25, 1000)
    dropblock_layer = tf.keras.Model.DropBlock2D(dropout_rate_scheduler=linear_scheduler)
    # dropblock layer will use linear scheduler as dropout rate
    ```
    """

    def __init__(self, initial_val, end_val, total_steps):
        self.initial_val = initial_val
        self.end_val = end_val
        self.total_steps = total_steps

    def __call__(self, current_step):
        return (
            self.initial_val
            + current_step * (self.end_val - self.initial_val) / self.total_steps
        )

    def get_config(self):
        return {
            "initial_val": self.initial_val,
            "end_val": self.end_val,
            "total_steps": self.total_steps,
        }
