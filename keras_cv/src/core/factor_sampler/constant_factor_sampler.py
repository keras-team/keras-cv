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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.core.factor_sampler.factor_sampler import FactorSampler


@keras_cv_export("keras_cv.core.ConstantFactorSampler")
class ConstantFactorSampler(FactorSampler):
    """ConstantFactorSampler samples the same factor for every call to
    `__call__()`.

    This is useful in cases where a user wants to always ensure that an
    augmentation layer performs augmentations of the same strength.

    Args:
        value: the value to return from `__call__()`.

    Example:
    ```python
    constant_factor = keras_cv.src.ConstantFactorSampler(0.5)
    random_sharpness = keras_cv.layers.RandomSharpness(factor=constant_factor)
    # random_sharpness will now always use a factor of 0.5
    ```
    """

    def __init__(self, value):
        self.value = value

    def __call__(self, shape=(), dtype="float32"):
        return tf.ones(shape=shape, dtype=dtype) * self.value

    def get_config(self):
        return {"value": self.value}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
