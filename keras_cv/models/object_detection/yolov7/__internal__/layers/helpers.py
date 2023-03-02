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


class ReOrganise(tf.keras.layers.Layer):
    """Reorganises the input from (b, w, h, c) to (b, w/2, h/2, 4c)."""

    def __init__(self, **kwargs):
        super(ReOrganise, self).__init__(**kwargs)

    def call(self, x):
        if len(x.shape) < 3 or len(x.shape) > 4:
            raise ValueError(
                "ReOrganise expects dimension of inputs to be either 3 or"
                f" 4. Received len(x.shape)={len(x.shape)}. "
                f"Expected either len(x.shape)=3 or len(x.shape)=3."
            )
        x = tf.concat(
            [
                x[..., ::2, ::2, :],
                x[..., 1::2, ::2, :],
                x[..., ::2, 1::2, :],
                x[..., 1::2, 1::2, :],
            ],
            -1,
        )
        return x


class Shortcut(tf.keras.layers.Layer):
    """Merges tensors along 0th dimension."""

    def __init__(self, **kwargs):
        super(Shortcut, self).__init__(**kwargs)

    def call(self, x):
        if len(x.shape) == 0:
            raise ValueError(
                "Shortcut expects dimension of inputs to be greater than"
                f" 0. Received len(x.shape)={len(x.shape)}."
                " Expected either len(x.shape)>0."
            )
        return tf.concat(x, axis=0)


class DownC(tf.keras.layers.Layer):
    """Spatial Pyramidal Pooling Layer. Used in YOLOv3-SPP"""

    def __init__(self, filters, n=1, kernel_size=2, **kwargs):
        super(DownC, self).__init__(**kwargs)
        self.filters = filters
        self.n = n
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.cv1 = FusedConvolution(input_shape[-1], 1, 1)
        self.cv2 = FusedConvolution(self.filters // 2, 3, self.kernel_size)
        self.cv3 = FusedConvolution(self.filters // 2, 1, 1)
        self.mp = tf.keras.layers.MaxPooling2D(
            pool_size=self.kernel_size, strides=self.kernel_size
        )

    def call(self, x):
        return tf.concat([self.cv2(self.cv1(x)), self.cv3(self.mp(x))], -1)

    def get_config(self):
        config = super(DownC, self).get_config()
        config.update(
            {"filters": self.filters, "n": self.n, "kernel_size": self.kernel_size}
        )
        return config


class FusedConvolution(tf.keras.layers.Layer):
    """Convolution and Batch Normalization operation fused together in
    order for faster processing during inference time. Training
    works similar normal convolution and then batch normalization operation."""

    def __init__(
        self,
        filters,
        kernel_size=1,
        strides=1,
        padding="same",
        groups=1,
        act=True,
        deploy=False,
        **kwargs,
    ):
        super(FusedConvolution, self).__init__(**kwargs)
        self.deploy = deploy
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.groups = groups
        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            groups=groups,
            use_bias=False,
        )
        self.bn = None
        if not deploy:
            self.bn = tf.keras.layers.BatchNormalization()
        self.act = (
            tf.keras.activations.swish
            if act is True
            else (act if isinstance(act, tf.keras.activations) else tf.identity)
        )

    def call(self, x):
        if self.deploy:
            return self.act(self.conv(x))
        else:
            return self.act(self.bn(self.conv(x)))

    def get_config(self):
        config = super(FusedConvolution, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "padding": self.padding,
                "strides": self.strides,
                "groups": self.groups,
            }
        )
        return config


class BreakDownFusedConv(tf.keras.layers.Layer):
    def __init__(self, depth=2, concat_dim=-1):
        super(BreakDownFusedConv, self).__init__()
        self.depth = depth
        self.part1 = None
        self.part2 = None
        self.concat_dim = concat_dim
        self.concat = tf.keras.layers.Concatenate(axis=concat_dim)

    def build(self, input_shape):
        channels = input_shape[-1]
        self.part1 = [
            FusedConvolution(channels / 2, kernel_size=1, strides=1)
            for _ in range(self.depth - 1)
        ]
        self.part1.append(FusedConvolution(channels / 2, kernel_size=3, strides=2))
        self.part2 = [tf.keras.layers.MaxPooling2D()]
        for i in range(self.depth - 1):
            self.part2.append(FusedConvolution(channels / 2, kernel_size=1, strides=1))

    def call(self, x):
        out = x
        x1 = out
        x2 = out
        for i in self.part1:
            x1 = i(x1)
        for i in self.part2:
            x2 = i(x2)
        return self.concat([x1, x2])

    def get_config(self):
        config = super(BreakDownFusedConv, self).get_config()
        config.update({"depth": self.depth, "concat_dim": self.concat_dim})
        return config


class Block(tf.keras.layers.Layer):
    def __init__(self, depth, block_depth, filters, filters2, concat_dim=-1):
        super(Block, self).__init__()
        self.depth = depth
        self.filters = filters
        self.filters2 = filters2
        self.block_depth = block_depth
        self.concat_dim = concat_dim
        self.part1 = FusedConvolution(filters, kernel_size=1, strides=1)
        self.part2 = [
            [
                FusedConvolution(filters, kernel_size=1, strides=1)
                if i == 0 and j == 0
                else FusedConvolution(filters, kernel_size=3, strides=1)
                for i in range(block_depth)
            ]
            for j in range(depth)
        ]
        self.part2.append([FusedConvolution(filters)])
        self.concat = tf.keras.layers.Concatenate(axis=concat_dim)
        self.fc1 = FusedConvolution(filters2, kernel_size=1, strides=1)

    def call(self, x):
        x1 = self.part1(x)
        x2 = []
        out = x
        for index, layer in enumerate(self.part2):
            for index2, layer2 in enumerate(layer):
                out = layer2(out)
                if index == len(self.part2) - 1:
                    continue
                elif index2 == 0:
                    x2.append(out)
        x2.append(out)
        fin = [x1]
        for i in x2:
            fin.append(i)
        return self.fc1(self.concat(fin))

    def get_config(self):
        config = super(Block, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "depth": self.depth,
                "block_depth": self.block_depth,
                "concat_dim": self.concat_dim,
            }
        )
        return config


class ImplicitAddition(tf.keras.layers.Layer):
    """Implicit Addition knowledge layer introduced in YOLOR."""

    def __init__(self, mean=0.0, std=0.02, **kwargs):
        super(ImplicitAddition, self).__init__(**kwargs)
        self.mean = mean
        self.std = std

    def build(self, input_shape):
        self.implicit = tf.Variable(
            initial_value=tf.random_normal_initializer(mean=self.mean, stddev=self.std)(
                shape=(1, 1, 1, input_shape[-1])
            ),
            trainable=True,
            name=self.name,
        )

    def call(self, x):
        return tf.cast(x, self.implicit.dtype) + self.implicit

    def get_config(self):
        config = super(ImplicitAddition, self).get_config()
        config.update({"mean": self.mean, "std": self.std})
        return config


class ImplicitMultiplication(tf.keras.layers.Layer):
    """Implicit Multiplication knowledge layer introduced in YOLOR."""

    def __init__(self, mean=1.0, std=0.02, **kwargs):
        super(ImplicitMultiplication, self).__init__(**kwargs)
        self.mean = mean
        self.std = std

    def build(self, input_shape):
        self.implicit = tf.Variable(
            initial_value=tf.random_normal_initializer(mean=self.mean, stddev=self.std)(
                shape=(1, 1, 1, input_shape[-1])
            ),
            trainable=True,
        )

    def call(self, x):
        return tf.cast(x, self.implicit.dtype) * self.implicit

    def get_config(self):
        config = super(ImplicitMultiplication, self).get_config()
        config.update({"mean": self.mean, "std": self.std})
        return config


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
    def _make_grid(nx=20, ny=20):
        grid_xy = tf.meshgrid(tf.range(ny), tf.range(nx))
        grid = tf.cast(tf.expand_dims(tf.stack(grid_xy, axis=-1), 2), tf.float32)
        return tf.cast(grid, tf.float32)


class IAuxDetect(tf.keras.layers.Layer):
    def __init__(self, nc=80, anchors=(), deploy=False, training=True, **kwargs):
        super(IAuxDetect, self).__init__(**kwargs)
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors) // 2
        self.deploy = deploy
        self.grid = [tf.zeros((1))] * self.nl
        a = tf.reshape(tf.cast(tf.constant(anchors), tf.float32), (self.nl, -1, 2))
        self.anchors = tf.Variable(a, trainable=False)
        self.anchor_grid = tf.Variable(
            tf.reshape(a, (self.nl, 1, 1, 1, -1, 2)), trainable=False
        )
        self.m = [tf.keras.layers.Conv2D(self.no * self.na, 1) for i in range(self.nl)]
        self.m2 = [tf.keras.layers.Conv2D(self.no * self.na, 1) for i in range(self.nl)]
        self.ia = [ImplicitAddition() for i in range(self.nl)]
        self.im = [ImplicitMultiplication() for i in range(self.nl)]
        self.training = training

    def call(self, x):
        z = []
        outputs = []
        aux_outputs = []
        for i in range(self.nl):
            if self.deploy:
                output = self.m[i](x[i])
            else:
                output = self.m[i](self.ia[i](x[i]))
                output = self.im[i](output)
            bs, ny, nx, _ = output.shape
            output = tf.reshape(output, (-1, ny, nx, self.na, self.no))
            outputs.append(output)
            if not self.deploy:
                aux_output = self.m2[i](x[i + self.nl])
                aux_output = tf.reshape(aux_output, (-1, ny, nx, self.na, self.no))
                aux_outputs.append(aux_output)

            if not self.training:
                if self.grid[i].shape[2:4] != output.shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny)
                y = tf.nn.sigmoid(output)
                xy, wg, conf = tf.split(y, (2, 2, self.nc + 1), axis=4)
                xy = y[..., :2] * (2.0 * self.stride[i]) + (
                    self.stride[i] * (self.grid[i] - 0.5)
                )
                wh = y[..., 2:4] ** 2 * (4 * self.anchor_grid[i])
                y = tf.concat((xy, wh, y[..., 4:]), axis=-1)
                z.append(tf.reshape(y, (-1, ny * nx * self.na, self.no)))

        return (
            (*outputs, *aux_outputs)
            if self.training
            else (*z, *[tf.zeros_like(z[0]) for i in range(len(z))])
        )

    def get_config(self):
        config = super(IAuxDetect, self).get_config()
        config.update(
            {
                "nc": self.nc,
                "anchors": self.anchors.numpy().reshape(self.nl, self.na * 2),
            }
        )
        return config

    def switch_to_deploy(self):
        for i in range(self.nl):
            kernel = tf.squeeze(self.m[i].kernel)
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


class SPPCSPC(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        n=1,
        groups=1,
        e=0.5,
        kernel_size=(5, 9, 13),
        deploy=False,
        **kwargs,
    ):
        super(SPPCSPC, self).__init__(**kwargs)
        c_ = int(2 * filters * e)
        self.filters = filter
        self.n = n
        self.groups = groups
        self.e = e
        self.kernel_size = kernel_size
        self.deploy = deploy
        self.cv1 = FusedConvolution(c_, 1, 1, deploy=deploy, groups=groups)
        self.cv2 = FusedConvolution(c_, 1, 1, deploy=deploy, groups=groups)
        self.cv3 = FusedConvolution(c_, 3, 1, deploy=deploy, groups=groups)
        self.cv4 = FusedConvolution(c_, 1, 1, deploy=deploy, groups=groups)
        self.cv5 = FusedConvolution(c_, 1, 1, deploy=deploy, groups=groups)
        self.cv6 = FusedConvolution(c_, 3, 1, deploy=deploy, groups=groups)
        self.cv7 = FusedConvolution(filters, 1, 1, deploy=deploy)
        self.m = [
            tf.keras.layers.MaxPooling2D(pool_size=n, strides=1, padding="same")
            for n in kernel_size
        ]

    def call(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        spp_output = tf.concat([x1] + [m(x) for m in self.m], axis=-1)
        csp_inp1 = self.cv6(self.cv5(spp_output))
        csp_inp2 = self.cv2(x)
        return self.cv7(tf.concat([csp_inp1, csp_inp2], axis=-1))

    def config(self):
        config = super(SPPCSPC, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "n": self.n,
                "groups": self.groups,
                "e": self.e,
                "kernel_size": self.kernel_size,
            }
        )
        return config
