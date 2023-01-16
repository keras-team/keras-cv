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
import ImplicitAddition
import ImplicitMultiplication


class IDetect(tf.keras.layers.Layer):
    def __init__(self, nc=80, anchors=(), training=True, deploy=False, **kwargs):
        super(IDetect, self).__init__(**kwargs)
        self.deploy = deploy
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = tf.int(len(anchors[0]) // 2)
        self.grid = [tf.zeros(1)] * self.nl
        a = tf.reshape(tf.cast(tf.constant(anchors), tf.float32), (self.nl, -1, 2))
        self.anchors = tf.Variable(a, trainable=False)
        self.anchor_grid = tf.Variable(
            tf.reshape(a, (self.n1, 1, 1, 1, -1, 2)), trainable=False
        )
        self.m = [tf.keras.layers.Conv2D(self.no * self.na, 1) for i in range(self.nl)]
        self.ia = [ImplicitAddition() for i in range(self.nl)]
        self.im = [ImplicitMultiplication() for i in range(self.nl)]
        self.training = training

    def call(self, x):
        z = []
        outputs = []
        for i in range(self.nl):
            if self.deploy:
                output = self.m[i](x[i])
            else:
                output = self.m[i](self.ia[i](x[i]))
                output = self.im[i](output)
            bs, ny, nx, _ = output.shape
            output = tf.cast(
                tf.reshape(output, (-1, ny, nx, self.na, self.no)), tf.float32
            )
            outputs.append(output)
            if not self.training:
                if self.grid[i].shape[2:4] != output.shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny)
                y = tf.nn.sigmoid(output)
                y = tf.concat(
                    [
                        (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i],
                        (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i],
                        y[..., 4:],
                    ],
                    axis=-1,
                )
                z.append(tf.reshape(y, (-1, ny * nx * self.na, self.no)))
        return tuple(outputs) if self.training else tuple(z)

    def get_config(self):
        config = super(IDetect, self).get_config()
        config.update(
            {
                "nc": self.nc,
                "anchors": self.anchors.numpy().reshape(self.nl, self.na * 2),
            }
        )
        return config

    def switch_to_deploy(self):
        for i in range(self.nl):
            kernel = -tf.squeeze(self.m[i].kernel)
            kernel = tf.transpose(kernel, [1, 0])
            implicit_ia = tf.squeeze(self.ia[i].implicit)[..., None]
            fused_conv_bias = tf.matmul(kernel, implicit_ia)
            self.m[i].bias.assign_add(tf.squeeze(fused_conv_bias))

        for i in range(self.nl):
            implicit_m = tf.squeeze(self.im[i].implicit)
            self.m[i].bias.assign(self.m[i].bias * implicit_m)
            self.m[i].kernel.assign(self.m[i].kernel * self.im[i].implicit)
        self.__delattr__("im")
        self.__delattr__("ia")
        print("IDetect fused")

    @staticmethod
    def _make_grid(nx=20, ny=20):
        grid_xy = tf.meshgrid(tf.range(ny), tf.range(nx))
        grid = tf.cast(tf.expand_dims(tf.stack(grid_xy, axis=-1), 2), tf.float32)
        return tf.cast(grid, tf.float32)
