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
from tensorflow.keras import backend

from keras_cv.core.factor_sampler.factor_sampler import Sampler


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class NormalFactorSampler(FactorSampler):
    """NormalFactorSampler samples factors from a normal distribution.

    This is useful in cases where a user wants to always ensure that an augmentation
    layer performs augmentations of the same strength.

    Args:
        mean: mean value for the distribution.
        standard_deviation: standard deviation of the distribution.
        lower: values below lower are clipped to lower.
        upper: values above upper are clipped to upper

    Usage:
    ```python
    factor = keras_cv.core.NormalFactor(mean=0.5, standard_deviation=0.1, lower=0, upper=1)
    random_sharpness = keras_cv.layers.RandomSharpness(factor=factor)
    # random_sharpness will now sample normally around 0.5, with a lower of 0 and upper
    # bound of 1.
    ```
    """

    def __init__(self, mean, standard_deviation, lower, upper, seed=None):
        self.mean = mean
        self.standard_deviation = standard_deviation
        self.lower = lower
        self.upper = upper
        self.seed = seed

    def sample(self):
        return tf.clip_by_value(
            tf.random.normal(
                (),
                mean=self.mean,
                stddev=self.standard_deviation,
                seed=self.seed,
                dtype=tf.float32,
            ),
            self.lower,
            self.upper,
        )

    def get_config(self):
        return {
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "lower": self.lower,
            "upper": self.upper,
            "seed": self.seed,
        }
