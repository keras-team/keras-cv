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
class FactorSampler:
    """FactorSampler represents a strength factor for use in an augmentation layer.

    FactorSampler should be subclassed and implement a `__call__()` method that returns
    a tf.float32, or a float.  This method will be used by preprocessing layers to
    determine the strength of their augmentation.  The specific range of values
    supported may vary by layer, but for most layers is the range [0, 1].
    """

    def __call__(self, shape=None, dtype="float32"):
        raise NotImplementedError(
            "FactorSampler subclasses must implement a `__call__()` method."
        )

    def get_config(self):
        return {}
