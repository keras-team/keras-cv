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

from keras_cv.core.factor.factor import Factor


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class UniformFactor(Factor):
    """UniformFactor samples factors uniformly from a range.

    This is useful in cases where a user wants to always ensure that an augmentation
    layer performs augmentations of the same strength.

    Args:
        lower: the lower bound of values returned from `sample()`.
        upper: the upper bound of values returned from `sample()`.

    Usage:
    ```python
    uniform_factor = keras_cv.core.UniformFactor(0, 0.5)
    random_sharpness = keras_cv.layers.RandomSharpness(factor=uniform_factor)
    # random_sharpness will now sample factors between 0, and 0.5
    ```
    """

    def __init__(self, lower, upper, seed=None, random_generator=None):
        self.lower = lower
        self.upper = upper

        if random_generator is None:
            random_generator = backend.RandomGenerator(seed, force_generator=False)
        self.random_generator = random_generator

    def sample(self):
        return self.random_generator.random_uniform(
            (), self.lower, self.upper, dtype=tf.float32
        )

    def get_config():
        return {
            "lower": self.lower,
            "upper": self.upper,
            "seed": self.seed,
            "random_generator": self.random_generator,
        }
