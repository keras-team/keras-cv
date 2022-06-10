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
class DropPath(tf.keras.layers.Layer):
    """
    Implements the DropPath layer. Some samples from the batch are randomly 
    dropped during training with dropping probability `drop_rate`. Note that 
    this layer DOES drop individual samples and not the entire batch. 
    
    Reference:
        - [FractalNet: Ultra-Deep Neural Networks without Residuals](https://arxiv.org/abs/1605.07648v4).
        - [rwightman/pytorch-image-models](https://tinyurl.com/timm-droppath)
    
    Args:
        drop_rate: float, the probability of the residual branch being dropped.
    
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
    def __init__(self, drop_rate=0.5):
        super().__init__()
        self.drop_rate = drop_rate

    def call(self, x, training=None):
        if self.drop_rate == 0. or not training:
            return x
        else:
            keep_prob = 1 - self.drop_rate
            drop_map_shape = (x.shape[0],) + (1,) * (len(x.shape) - 1)
            drop_map = tf.keras.backend.random_bernoulli(drop_map_shape,
                                                         p=keep_prob)
            x = x / keep_prob
            x = x * drop_map
            return x