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
import random as python_random

from keras_cv.src.backend import keras
from keras_cv.src.backend.config import keras_3

if keras_3():
    from keras.random import *  # noqa: F403, F401
else:
    from keras_core.random import *  # noqa: F403, F401


def _make_default_seed():
    return python_random.randint(1, int(1e9))


class SeedGenerator:
    def __new__(cls, seed=None, **kwargs):
        if keras_3():
            return keras.random.SeedGenerator(seed=seed, **kwargs)
        return super().__new__(cls)

    def __init__(self, seed=None):
        if seed is None:
            seed = _make_default_seed()
        self._initial_seed = seed
        self._current_seed = [0, seed]

    def next(self, ordered=True):
        self._current_seed[0] += 1
        return self._current_seed[:]

    def get_config(self):
        return {"seed": self._initial_seed}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _draw_seed(seed):
    if keras_3():
        # Keras 3 seed can be directly passed to random functions
        return seed
    if isinstance(seed, SeedGenerator):
        init_seed = seed.next()
    else:
        if seed is None:
            seed = _make_default_seed()
        init_seed = [0, seed]
    return init_seed


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    seed = _draw_seed(seed)
    kwargs = {}
    if dtype:
        kwargs["dtype"] = dtype
    if keras_3():
        return keras.random.normal(
            shape,
            mean=mean,
            stddev=stddev,
            seed=seed,
            **kwargs,
        )
    else:
        import tensorflow as tf

        return tf.random.stateless_normal(
            shape,
            mean=mean,
            stddev=stddev,
            seed=seed,
            **kwargs,
        )


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    init_seed = _draw_seed(seed)
    kwargs = {}
    if dtype:
        kwargs["dtype"] = dtype
    if keras_3():
        return keras.random.uniform(
            shape,
            minval=minval,
            maxval=maxval,
            seed=init_seed,
            **kwargs,
        )
    else:
        import tensorflow as tf

        return tf.random.stateless_uniform(
            shape,
            minval=minval,
            maxval=maxval,
            seed=init_seed,
            **kwargs,
        )


def shuffle(x, axis=0, seed=None):
    init_seed = _draw_seed(seed)
    if keras_3():
        return keras.random.shuffle(x=x, axis=axis, seed=init_seed)
    else:
        import tensorflow as tf

        return tf.random.stateless_shuffle(x=x, axis=axis, seed=init_seed)


def categorical(logits, num_samples, dtype=None, seed=None):
    init_seed = _draw_seed(seed)
    kwargs = {}
    if dtype:
        kwargs["dtype"] = dtype
    if keras_3():
        return keras.random.categorical(
            logits=logits,
            num_samples=num_samples,
            seed=init_seed,
            **kwargs,
        )
    else:
        import tensorflow as tf

        return tf.random.stateless_categorical(
            logits=logits,
            num_samples=num_samples,
            seed=init_seed,
            **kwargs,
        )
