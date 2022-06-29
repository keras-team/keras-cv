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

from keras_cv.core.factor_sampler.factor_sampler import FactorSampler


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class UniformFactorSampler(FactorSampler):
    """UniformFactorSampler samples factors uniformly from a range.

    This is useful in cases where a user wants to always ensure that an augmentation
    layer performs augmentations of the same strength.

    Args:
        lower: the lower bound of values returned from `__call__()`.
        upper: the upper bound of values returned from `__call__()`.
        seed: A shape int or Tensor, the seed to the random number generator. Must have
            dtype int32 or int64. (When using XLA, only int32 is allowed.)
    Usage:
    ```python
    uniform_factor = keras_cv.UniformFactorSampler(0, 0.5)
    random_sharpness = keras_cv.layers.RandomSharpness(factor=uniform_factor)
    # random_sharpness will now sample factors between 0, and 0.5
    ```
    """

    def __init__(self, lower, upper, seed=None):
        self.lower = lower
        self.upper = upper
        self.seed = seed

    def __call__(self, shape=None, dtype="float32"):
        shape = shape or ()
        return tf.random.uniform(
            shape, seed=self.seed, minval=self.lower, maxval=self.upper, dtype=dtype
        )

    def get_config(self):
        return {
            "lower": self.lower,
            "upper": self.upper,
            "seed": self.seed,
        }
