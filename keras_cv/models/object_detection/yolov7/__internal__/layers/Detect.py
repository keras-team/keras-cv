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


class Detect(tf.keras.layers.Layer):
    def __init__(self, nc=80, anchors=(), training=True, **kwargs):
        super(Detect, self).__init__(**kwargs)
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [tf.zeros(1)] * self.nl
        a = tf.cast(tf.reshape(tf.constant(anchors), (self.nl, -1, 2)), tf.float32)
        self.anchors = tf.Variable(a, trainable=False)
        self.anchor_grid = tf.Variable(
            tf.reshape(a, (self.nl, 1, 1, 1, -1, 2)), trainable=False
        )
        self.m = [tf.keras.layers.Conv2D(self.no * self.na, 1) for _ in range(self.nl)]
        self.training = training

    def call(self, x):
        z = []
        outputs = []
        for i in range(self.nl):
            output = self.m[i](x[i])
            bs, ny, nx, _ = x[i].shape
            output = tf.reshape(output, (-1, ny, nx, self.na, self.no))
            outputs.append(output)
            if not self.training:
                if self.grid[i].shape[1:3] != x[i].shape[1:3]:
                    self.grid[i] = self._make_grid(nx, ny)
                y = tf.nn.sigmoid(output)
                y = tf.concat(
                    [
                        (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i],  # xy
                        (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i],
                        y[..., 4:],
                    ],
                    axis=-1,
                )
                z.append(tf.reshape(y, (-1, ny * nx * self.na, self.no)))
        if self.training:
            return tuple(outputs)
        else:
            return tuple(z)

    @staticmethod
    def _make_grid(nx=220, ny=20):
        grid_xy = tf.meshgrid(tf.range(ny), tf.range(nx))
        grid = tf.cast(tf.expand_dims(tf.stack(grid_xy, axis=-1), 2), tf.float32)
        return tf.cast(grid, tf.float32)
