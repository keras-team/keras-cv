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


@keras_cv_export("keras_cv.core.NormalFactorSampler")
class NormalFactorSampler(FactorSampler):
    """NormalFactorSampler samples factors from a normal distribution.

    This is useful in cases where a user wants to always ensure that an
    augmentation layer performs augmentations of the same strength.

    Args:
        mean: mean value for the distribution.
        stddev: standard deviation of the distribution.
        min_value: values below min_value are clipped to min_value.
        max_value: values above max_value are clipped to max_value.

    Example:
    ```python
    factor = keras_cv.core.NormalFactor(
        mean=0.5,
        stddev=0.1,
        lower=0,
        upper=1
    )
    random_sharpness = keras_cv.layers.RandomSharpness(factor=factor)
    # random_sharpness will now sample normally around 0.5, with a lower of 0
    # and upper bound of 1.
    ```
    """

    def __init__(self, mean, stddev, min_value, max_value, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.min_value = min_value
        self.max_value = max_value
        self.seed = seed

    def __call__(self, shape=(), dtype="float32"):
        return tf.clip_by_value(
            tf.random.normal(
                shape=shape,
                mean=self.mean,
                stddev=self.stddev,
                seed=self.seed,
                dtype=dtype,
            ),
            self.min_value,
            self.max_value,
        )

    def get_config(self):
        return {
            "mean": self.mean,
            "stddev": self.stddev,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "seed": self.seed,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
