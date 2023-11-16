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
from keras_cv.backend import keras
from keras_cv.backend.config import keras_3

if keras_3():
    from keras.random import *  # noqa: F403, F401
    # SeedGenerator is imported from `keras.random`
else:
    from keras_core.random import *  # noqa: F403, F401
    class SeedGenerator:
        def __init__(self, seed=None, **kwargs):
            self._current_seed = [seed, 0]

        def next(self, ordered=True):
            self._current_seed[1] += 1
            return self._current_seed[:]


def make_seed(seed=None):
    if isinstance(seed, SeedGenerator):
        seed_0, seed_1 = seed.next()
        if seed_0 is None:
            init_seed = seed_1
        else:
            init_seed = seed_0 + seed_1
    else:
        init_seed = seed
    return init_seed


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
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

        return tf.random.normal(
            shape,
            mean=mean,
            stddev=stddev,
            seed=make_seed(seed),
            **kwargs,
        )


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    kwargs = {}
    if dtype:
        kwargs["dtype"] = dtype
    if keras_3():
        return keras.random.uniform(
            shape,
            minval=minval,
            maxval=maxval,
            seed=seed,
            **kwargs,
        )
    else:
        import tensorflow as tf

        return tf.random.uniform(
            shape,
            minval=minval,
            maxval=maxval,
            seed=make_seed(seed),
            **kwargs,
        )
    
def randint(shape, minval=0.0, maxval=1.0, dtype="int32", seed=None):
    kwargs = {}
    if dtype:
        kwargs["dtype"] = dtype
    if keras_3():
        return keras.random.randint(
            shape,
            minval=minval,
            maxval=maxval,
            seed=seed,
            **kwargs,
        )
    else:
        import tensorflow as tf

        return tf.random.uniform(
            shape,
            minval=minval,
            maxval=maxval,
            seed=make_seed(seed),
            **kwargs,
        )


def shuffle(x, axis=0, seed=None):
    init_seed = make_seed(seed)
    if keras_3():
        return keras.random.shuffle(x=x, axis=axis, seed=init_seed)
    else:
        import tensorflow as tf

        return tf.random.shuffle(x=x, axis=axis, seed=init_seed)


def categorical(logits, num_samples, dtype=None, seed=None):
    init_seed = make_seed(seed)
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

        return tf.random.categorical(
            logits=logits,
            num_samples=num_samples,
            seed=init_seed,
            **kwargs,
        )
