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
class StochasticDepth(tf.keras.layers.Layer):
    """
    Implements the Stochastic Depth layer. It randomly drops residual branches
    in residual architectures. It is used as a drop-in replacement for addition
    operation. Note that this layer DOES NOT drop a residual block across
    individual samples but across the entire batch.

    Reference:
        - [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
        - Docstring taken from [stochastic_depth.py](https://tinyurl.com/mr3y2af6)

    Args:
        rate: float, the probability of the residual branch being dropped.

    Usage:

    `StochasticDepth` can be used in a residual network as follows:
    ```python
    # (...)
    input = tf.ones((1, 3, 3, 1), dtype=tf.float32)
    residual = tf.keras.layers.Conv2D(1, 1)(input)
    output = keras_cv.layers.StochasticDepth()([input, residual])
    # (...)
    ```

    At train time, StochasticDepth returns:
    $$
    x[0] + b_l * x[1],
    $$
    where $b_l$ is a random Bernoulli variable with probability
    $P(b_l = 1) = rate$. At test time, StochasticDepth rescales the activations
    of the residual branch based on the drop rate ($rate$):
    $$
    x[0] + (1 - rate) * x[1]
    $$
    """

    def __init__(self, rate=0.5, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.survival_probability = 1.0 - self.rate

    def call(self, x, training=None):
        if len(x) != 2:
            raise ValueError(
                f"""Input must be a list of length 2. """
                f"""Got input with length={len(x)}."""
            )

        shortcut, residual = x

        b_l = tf.keras.backend.random_bernoulli([], p=self.survival_probability)

        if training:
            return shortcut + b_l * residual
        else:
            return shortcut + self.survival_probability * residual

    def get_config(self):
        config = {"rate": self.rate}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
