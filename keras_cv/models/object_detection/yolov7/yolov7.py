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
from keras_cv.models.object_detection.yolov7.__internal__.layers import (
    layers as layer_lib,
)


def FusedLayer(downC_filters, filters, filters2, x):
    out = layer_lib.DownC(downC_filters)(x)

    # Part 1
    out1 = layer_lib.FusedConvolution(filters)(out)

    # Part 2
    out2 = layer_lib.FusedConvolution(filters)(out)

    x1 = layer_lib.FusedConvolution(filters)(layer_lib.FusedConvolution(filters)(out2))
    x2 = layer_lib.FusedConvolution(filters)(layer_lib.FusedConvolution(filters)(x1))
    x3 = layer_lib.FusedConvolution(filters)(layer_lib.FusedConvolution(filters)(x2))

    part1Out = layer_lib.FusedConvolution(filters2)(
        tf.keras.layers.Concatenate()([out2, x1, x2, x3, out1])
    )

    # Part 3
    out3 = layer_lib.FusedConvolution(filters)(out)

    # Part 4
    out4 = layer_lib.FusedConvolution(filters)(out)
    x1 = layer_lib.FusedConvolution(filters)(layer_lib.FusedConvolution(filters)(out4))
    x2 = layer_lib.FusedConvolution(filters)(layer_lib.FusedConvolution(filters)(x1))
    x3 = layer_lib.FusedConvolution(filters)(layer_lib.FusedConvolution(filters)(x2))

    part2Out = layer_lib.FusedConvolution(filters2)(
        tf.keras.layers.Concatenate()([out3, out4, x1, x2, x3])
    )

    return layer_lib.Shortcut()([part1Out, part2Out])


def UpsampleFusedLayer(filters, filters2, filters3, x):
    first, second = x[0], x[1]
    out = layer_lib.FusedConvolution(filters2)(first)
    out = tf.keras.layers.UpSampling2D(size=2, interpolation="nearest")(out)
    out = tf.keras.layers.Concatenate()([out, second])
    # Part1
    out1 = layer_lib.FusedConvolution(filters3)(out)

    # Part2
    out2 = layer_lib.FusedConvolution(filters3)(out)
    x1 = layer_lib.FusedConvolution(filters)(out2)
    x2 = layer_lib.FusedConvolution(filters)(x1)
    x3 = layer_lib.FusedConvolution(filters)(x2)
    x4 = layer_lib.FusedConvolution(filters)(x3)
    x5 = layer_lib.FusedConvolution(filters)(x4)
    x6 = layer_lib.FusedConvolution(filters)(x5)

    part1Out = layer_lib.FusedConvolution(filters2)(
        tf.keras.layers.Concatenate()([out1, out2, x1, x2, x3, x4, x5, x6])
    )

    # Part3
    out3 = layer_lib.FusedConvolution(filters3)(out)

    # Part4
    out4 = layer_lib.FusedConvolution(filters3)(out)
    x1 = layer_lib.FusedConvolution(filters)(out4)
    x2 = layer_lib.FusedConvolution(filters)(x1)
    x3 = layer_lib.FusedConvolution(filters)(x2)
    x4 = layer_lib.FusedConvolution(filters)(x3)
    x5 = layer_lib.FusedConvolution(filters)(x4)
    x6 = layer_lib.FusedConvolution(filters)(x5)

    part2Out = layer_lib.FusedConvolution(filters2)(
        tf.keras.layers.Concatenate()([out3, out4, x1, x2, x3, x4, x5, x6])
    )

    out = layer_lib.Shortcut()([part1Out, part2Out])
    return out, part2Out


def FusedLayer2(filters, filters2, filters3, x):
    first, second = x[0], x[1]
    out = layer_lib.DownC(filters3)(first)
    out = tf.keras.layers.Concatenate()([out, second])
    # Part1
    out1 = layer_lib.FusedConvolution(filters3)(out)

    # Part 2
    out2 = layer_lib.FusedConvolution(filters3)(out)
    x1 = layer_lib.FusedConvolution(filters)(out2)
    x2 = layer_lib.FusedConvolution(filters)(x1)
    x3 = layer_lib.FusedConvolution(filters)(x2)
    x4 = layer_lib.FusedConvolution(filters)(x3)
    x5 = layer_lib.FusedConvolution(filters)(x4)
    x6 = layer_lib.FusedConvolution(filters)(x5)

    part1Out = layer_lib.FusedConvolution(filters2)(
        tf.keras.layers.Concatenate()([out1, out2, x1, x2, x3, x4, x5, x6])
    )

    # Part 3
    out3 = layer_lib.FusedConvolution(filters3)(out)

    # Part 4
    out4 = layer_lib.FusedConvolution(filters3)(out)
    x1 = layer_lib.FusedConvolution(filters)(out4)
    x2 = layer_lib.FusedConvolution(filters)(x1)
    x3 = layer_lib.FusedConvolution(filters)(x2)
    x4 = layer_lib.FusedConvolution(filters)(x3)
    x5 = layer_lib.FusedConvolution(filters)(x4)
    x6 = layer_lib.FusedConvolution(filters)(x5)

    part2Out = layer_lib.FusedConvolution(filters2)(
        tf.keras.layers.Concatenate()([out3, out4, x1, x2, x3, x4, x5, x6])
    )
    out = layer_lib.Shortcut()([part1Out, part2Out])
    return out


def YOLOV7_e6e(input_shape=(640, 640, 3)):
    input = tf.keras.layers.InputLayer(input_shape=input_shape)
    reorg = layer_lib.ReOrg()(input.input)
    fc1 = layer_lib.FusedConvolution(80)(reorg)
    config_down = [
        (160, 64, 160, False),  # No residual shortcut
        (320, 128, 320, True),  # No Residual shortcut
        (640, 256, 640, True),  # Residual
        (960, 384, 960, True),  # Residual
        (1280, 512, 1280, True),  # Residual
    ]
    config_up = [
        (192, 480, 384, True),  # Concate shortcut
        (128, 320, 256, True),  # Concate shortcut
        (64, 160, 128, True),  # Concate shortcut
    ]
    config_second = [
        (128, 320, 256, True, True),  # Concate and residual shortcut
        (192, 480, 384, True, True),  # Concate and residual shortcut
        (256, 640, 512, True, False),  # Concate and residual shortcut
    ]
    anchors = [
        [19, 27, 44, 40, 38, 94],
        [96, 68, 86, 152, 180, 137],
        [140, 301, 303, 264, 238, 542],
        [436, 615, 739, 380, 925, 792],
    ]
    nc = 80
    aux_input = []
    # Downsample part
    out = fc1
    inter1 = []
    for index, item in enumerate(config_down):
        downc_filters, filters, filters2, residual = item
        out = FusedLayer(downc_filters, filters, filters2, out)
        if residual:
            channels = out.shape[-1]
            inter1.append(layer_lib.FusedConvolution(channels // 2)(out))

    inter1.pop()
    # SPPCSP Net
    sppcspc = layer_lib.SPPCSPC(640)(out)

    out = sppcspc
    inter2 = []
    shortcut_inter = [sppcspc]
    # Upsample Part
    for index, item in enumerate(config_up):
        filters, filters2, filters3, concatenate = item
        if concatenate:
            inp = [out]
            inp.append(inter1.pop())
        out, fused = UpsampleFusedLayer(filters, filters2, filters3, inp)
        channels = fused.shape[-1]
        inter2.append(layer_lib.FusedConvolution(channels * 2)(fused))
        if index == len(config_up) - 1:
            channels = out.shape[-1]
            aux_input.append(layer_lib.FusedConvolution(channels * 2)(out))
        else:
            shortcut_inter.append(out)

    # Second Part
    shortcut_inter2 = []
    for index, item in enumerate(config_second):
        filters, filters2, filters3, concatenate, residual = item
        if concatenate:
            inp = [out]
            inp.append(shortcut_inter.pop())
        out = FusedLayer2(filters, filters2, filters3, inp)
        if residual:
            channels = out.shape[-1]
            shortcut_inter2.append(layer_lib.FusedConvolution(channels * 2)(out))
    for i in inter2:
        aux_input.append(i)
    for i in shortcut_inter2:
        aux_input.append(i)
    aux_input.append(layer_lib.FusedConvolution(2 * out.shape[-1])(out))
    aux_input.append(layer_lib.FusedConvolution(2 * sppcspc.shape[-1])(sppcspc))
    out = layer_lib.IAuxDetect(nc=nc, anchors=anchors)(aux_input)
    model = tf.keras.models.Model(inputs=input.input, outputs=out)
    return model
