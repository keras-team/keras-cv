# Copyright 2022 The KerasCV Authors. All Rights Reserved.
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

from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.models.stable_diffusion.padded_conv2d import PaddedConv2D


class AttentionBlock(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.norm = keras.layers.GroupNormalization(epsilon=1e-5)
        self.q = PaddedConv2D(output_dim, 1)
        self.k = PaddedConv2D(output_dim, 1)
        self.v = PaddedConv2D(output_dim, 1)
        self.proj_out = PaddedConv2D(output_dim, 1)

    def call(self, inputs):
        x = self.norm(inputs)
        q, k, v = self.q(x), self.k(x), self.v(x)

        # Compute attention
        shape = ops.shape(q)
        h, w, c = shape[1], shape[2], shape[3]
        q = ops.reshape(q, (-1, h * w, c))  # b, hw, c
        k = ops.transpose(k, (0, 3, 1, 2))
        k = ops.reshape(k, (-1, c, h * w))  # b, c, hw
        y = q @ k
        y = y * 1 / ops.sqrt(ops.cast(c, self.compute_dtype))
        y = keras.activations.softmax(y)

        # Attend to values
        v = ops.transpose(v, (0, 3, 1, 2))
        v = ops.reshape(v, (-1, c, h * w))
        y = ops.transpose(y, (0, 2, 1))
        x = v @ y
        x = ops.transpose(x, (0, 2, 1))
        x = ops.reshape(x, (-1, h, w, c))
        return self.proj_out(x) + inputs
