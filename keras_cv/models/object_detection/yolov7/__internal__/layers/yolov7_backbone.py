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
from helpers import FusedConvolution, Shortcut, DownC


class Block(tf.keras.layers.Layer):
    def __init__(self, depth, block_depth, filters, concat_dim=-1):
        super(Block, self).__init__()
        self.depth = depth
        self.filters = filters
        self.block_depth = block_depth
        self.concat_dim = concat_dim
        self.part1 = FusedConvolution(filters)
        self.part2 = [
            [FusedConvolution(filters) for i in range(block_depth)]
            for j in range(depth - 1)
        ]
        self.part2.append([FusedConvolution(filters)])
        self.concat = tf.keras.layers.Concatenate(axis=concat_dim)
        self.fc1 = FusedConvolution((filters * (depth + 2)) / 2)

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
        res = self.fc1(self.concat(fin))
        return res

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


class YOLOV7BackBone(tf.keras.layers.Layer):
    def __init__(
        self, filters, num_blocks, depth, block_depth, concat_dim=-1, width_multiplier=1
    ):
        super(YOLOV7BackBone, self).__init__()
        self.filters = filters
        self.num_blocks = num_blocks
        self.depth = depth
        self.block_depth = block_depth
        self.width_multiplier = width_multiplier
        self.b = [
            [
                Block(depth, block_depth, filter, concat_dim)
                for _ in range(width_multiplier)
            ]
            for filter in filters
        ]

    def call(self, x):
        out = x
        for i in self.b:
            if len(i) == 1:
                out = i[0](out)
            else:
                channels = out.shape[-1]
                out = DownC(channels * 2)(out)
                inter = [layer(out) for layer in i]
                out = Shortcut()(inter)
        return out

    def get_config(self):
        config = super(YOLOV7BackBone, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "num_blocks": self.num_blocks,
                "depth": self.depth,
                "block_depth": self.block_depth,
                "concat_dim": self.concat_dim,
                "width_multiplier": self.width_multiplier,
            }
        )
        return config
