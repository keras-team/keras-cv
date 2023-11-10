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

from keras_cv.backend.config import keras_3
from keras_cv.backend import keras

if keras_3():
    from keras.random import *  # noqa: F403, F401
else:
    from keras_core.random import *  # noqa: F403, F401


class RandomGenerator:
    """Random generator that selects appropriate random ops.
    
    Created for compatibility between Keras 2 and Keras 3.
    """

    def __init__(self, seed=None, **kwargs):
        self._seed = seed
        self._is_keras_3 = keras_3()
        if self._is_keras_3:
            # Ignore args specifc to keras 2 RandomGenerator
            kwargs.pop("force_generator", None)
            kwargs.pop("rng_type", None)
            self._seed_gen = keras.random.SeedGenerator(seed, **kwargs) 
        else:
            self._v2_random_gen = keras.backend.RadomGenerator(seed, **kwargs)

    def random_normal(self, shape, mean=0.0, stddev=1.0, dtype=None):
        """Produce random number based on the normal distribution.
        """
        if self._is_keras_3:
            return keras.random.normal(shape, mean=mean, stddev=stddev, dtype=dtype, seed=self._seed_gen)
        else:
            return self._v2_random_gen.random_normal(shape, mean=mean, stddev=stddev, dtype=dtype)

    def random_uniform(
        self, shape, minval=0.0, maxval=None, dtype=None
    ):
        """Produce random number based on the uniform distribution."""
        if self._is_keras_3:
            return keras.random.uniform(shape, minval=minval, maxval=maxval, dtype=dtype, seed=self._seed_gen)
        else:
            return self._v2_random_gen.random_uniform(shape, minval=minval, maxval=maxval, dtype=dtype)


    def truncated_normal(
        self, shape, mean=0.0, stddev=1.0, dtype=None
    ):
        """Produce random number based on the truncated normal distribution."""
        if self._is_keras_3:
            return keras.random.truncated_normal(shape, mean=mean, stddev=stddev, dtype=dtype, seed=self._seed_gen)
        else:
            return self._v2_random_gen.truncated_normal(shape, mean=mean, stddev=stddev, dtype=dtype)


    def make_legacy_seed(self):
        """Create a new seed for the legacy stateful ops to use."""

        if self._is_keras_3:
            return self._seed_gen.next()
        else:
            return self._v2_random_gen.make_legacy_seed()
