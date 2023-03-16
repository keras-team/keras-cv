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


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class DropPath(tf.keras.__internal__.layers.BaseRandomLayer):
    """
    Implements the DropPath layer. DropPath randomly drops samples during training
     with a probability of `rate`. Note that this layer drops individual samples
    within a batch and not the entire batch. DropPath randomly drops some of the
    individual samples from a batch, whereas StachasticDepth randomly drops the
    entire batch.

    References:
        - [FractalNet](https://arxiv.org/abs/1605.07648v4).
        - [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models/blob/7c67d6aca992f039eece0af5f7c29a43d48c00e4/timm/models/layers/drop.py#L135)

    Args:
        rate: float, the probability of the residual branch being dropped.
        seed: (Optional) Integer. Used to create a random seed.

    Usage:
    `DropPath` can be used in any network as follows:
    ```python

    # (...)
    input = tf.ones((1, 3, 3, 1), dtype=tf.float32)
    residual = tf.keras.layers.Conv2D(1, 1)(input)
    output = keras_cv.layers.DropPath()(input)
    # (...)
    ```
    """

    def __init__(self, rate=0.5, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.rate = rate
        self.seed = seed

    def call(self, x, training=None):
        if self.rate == 0.0 or not training:
            return x
        else:
            keep_prob = 1 - self.rate
            drop_map_shape = (x.shape[0],) + (1,) * (len(x.shape) - 1)
            drop_map = tf.keras.backend.random_bernoulli(
                drop_map_shape, p=keep_prob, seed=self.seed
            )
            x = x / keep_prob
            x = x * drop_map
            return x

    def get_config(self):
        config = {"rate": self.rate, "seed": self.seed}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
