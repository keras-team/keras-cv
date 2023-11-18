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

import random

import keras

if hasattr(keras, "src"):
    keras_backend = keras.src.backend
else:
    keras_backend = keras.backend

from keras_cv.backend import keras
from keras_cv.backend.config import keras_3

if keras_3():
    from keras.random import *  # noqa: F403, F401
else:
    from keras_core.random import *  # noqa: F403, F401


class RandomGenerator:
    """Random generator that selects appropriate random ops.

    Used for Keras 2 and Keras 3 compatibilty.

    Replace with `SeedGenerator` when dropping Keras 2 compatibility.
    """

    def __call__(cls, *args, **kwargs):
        if keras_3():
            return cls.__new__(*args, **kwargs)
        else:
            return keras_backend.RandomGenerator(*args, **kwargs)

    def __init__(self, seed=None, force_generator=None):
        self._seed = self._create_seed(seed)

    def _create_seed(self, user_specified_seed):
        if user_specified_seed is not None:
            return user_specified_seed
        else:
            return random.randint(1, int(1e9))

    def make_legacy_seed(self):
        """Create a new seed for the legacy stateful ops to use.

        Returns:
          int as new seed, or None.
        """
        if self._seed is not None:
            result = self._seed
            self._seed += 1
            return result
        return None

    def random_normal(
        self, shape, mean=0.0, stddev=1.0, dtype=None, nonce=None
    ):
        """Produce random number based on the normal distribution."""
        import tensorflow as tf

        dtype = dtype or "float32"
        return tf.random.normal(
            shape=shape,
            mean=mean,
            stddev=stddev,
            dtype=dtype,
            seed=self.make_legacy_seed(),
        )

    def random_uniform(
        self, shape, minval=0.0, maxval=None, dtype=None, nonce=None
    ):
        """Produce random number based on the uniform distribution."""
        import tensorflow as tf

        dtype = dtype or "float32"
        return tf.random.uniform(
            shape=shape,
            minval=minval,
            maxval=maxval,
            dtype=dtype,
            seed=self.make_legacy_seed(),
        )
