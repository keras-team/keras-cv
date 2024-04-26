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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.backend import random


@keras_cv_export("keras_cv.layers.DropPath")
class DropPath(keras.layers.Layer):
    """
    Implements the DropPath layer. DropPath randomly drops samples during
    training with a probability of `rate`. Note that this layer drops individual
    samples within a batch and not the entire batch. DropPath randomly drops
    some individual samples from a batch, whereas StochasticDepth
    randomly drops the entire batch.

    References:
        - [FractalNet](https://arxiv.org/abs/1605.07648v4).
        - [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models/blob/7c67d6aca992f039eece0af5f7c29a43d48c00e4/timm/models/layers/drop.py#L135)

    Args:
        rate: float, the probability of the residual branch being dropped.
        seed: (Optional) integer. Used to create a random seed.

    Example:
    `DropPath` can be used in any network as follows:
    ```python

    # (...)
    input = tf.ones((1, 3, 3, 1), dtype=tf.float32)
    residual = keras.layers.Conv2D(1, 1)(input)
    output = keras_cv.layers.DropPath()(input)
    # (...)
    ```
    """  # noqa: E501

    def __init__(self, rate=0.5, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self._seed_val = seed
        self.seed = random.SeedGenerator(seed=seed)

    def call(self, x, training=None):
        if self.rate == 0.0 or not training:
            return x
        else:
            batch_size = x.shape[0] or ops.shape(x)[0]
            drop_map_shape = (batch_size,) + (1,) * (len(x.shape) - 1)
            drop_map = ops.cast(
                random.uniform(drop_map_shape, seed=self.seed) > self.rate,
                x.dtype,
            )
            x = x / (1.0 - self.rate)
            x = x * drop_map
            return x

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate, "seed": self._seed_val})
        return config
