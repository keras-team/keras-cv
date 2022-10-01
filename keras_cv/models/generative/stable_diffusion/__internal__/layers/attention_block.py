# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow import keras

from keras_cv.models.generative.stable_diffusion.__internal__.layers.group_normalization import (
    GroupNormalization,
)
from keras_cv.models.generative.stable_diffusion.__internal__.layers.padded_conv2d import (
    PaddedConv2D,
)


class AttentionBlock(keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.norm = GroupNormalization(epsilon=1e-5)
        self.q = PaddedConv2D(channels, 1)
        self.k = PaddedConv2D(channels, 1)
        self.v = PaddedConv2D(channels, 1)
        self.proj_out = PaddedConv2D(channels, 1)

    def call(self, x):
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)

        # Compute attention
        b, h, w, c = q.shape
        q = tf.reshape(q, (-1, h * w, c))  # b,hw,c
        k = keras.layers.Permute((3, 1, 2))(k)
        k = tf.reshape(k, (-1, c, h * w))  # b,c,hw
        w_ = q @ k
        w_ = w_ * (c ** (-0.5))
        w_ = keras.activations.softmax(w_)

        # Attend to values
        v = keras.layers.Permute((3, 1, 2))(v)
        v = tf.reshape(v, (-1, c, h * w))
        w_ = keras.layers.Permute((2, 1))(w_)
        h_ = v @ w_
        h_ = keras.layers.Permute((2, 1))(h_)
        h_ = tf.reshape(h_, (-1, h, w, c))
        return x + self.proj_out(h_)
