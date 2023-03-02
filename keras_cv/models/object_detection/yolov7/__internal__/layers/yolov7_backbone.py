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
from helpers import (
    FusedConvolution,
    Shortcut,
    DownC,
    ReOrganise,
    Block,
    BreakDownFusedConv,
)

BACKBONE_MODEL_CONFIGS = {
    "YOLOv7-d6": {
        "initial_filters": [{"filters": 96, "kernel_size": 3, "strides": 1}],
        "filters": [64, 128, 256, 384, 512],
        "filters2": [192, 384, 768, 1152, 1536],
        "num_blocks": 5,
        "depth": 4,
        "block_depth": 2,
        "downsampling": [True for _ in range(5)],
        "downsampling_type": ["downc" for _ in range(5)],
        "reorganise": True,
        "concat_dim": 1,
        "width_multiplier": 1,
    },
    "YOLOv7-e6": {
        "initial_filters": [{"filters": 80, "kernel_size": 3, "strides": 1}],
        "filters": [64, 128, 256, 384, 512],
        "filters2": [160, 320, 640, 960, 1280],
        "num_blocks": 5,
        "depth": 3,
        "block_depth": 2,
        "downsampling": [True for _ in range(5)],
        "downsampling_type": ["downc" for _ in range(5)],
        "reorganise": True,
        "concat_dim": 1,
        "width_multiplier": 1,
    },
    "YOLOv7-e6e": {
        "initial_filters": [{"filters": 80, "kernel_size": 3, "strides": 1}],
        "filters": [64, 128, 256, 384, 512],
        "filters2": [160, 320, 640, 960, 1280],
        "num_blocks": 5,
        "depth": 3,
        "block_depth": 2,
        "downsampling": [True for _ in range(5)],
        "downsampling_type": ["downc" for _ in range(5)],
        "reorganise": True,
        "concat_dim": -1,
        "width_multiplier": 2,
    },
    "YOLOv7-tiny": {
        "initial_filters": [{"filters": 32, "kernel_size": 3, "strides": 2}],
        "filters": [32, 64, 128, 256],
        "filters2": [64, 128, 256, 512],
        "num_blocks": 4,
        "depth": 2,
        "block_depth": 1,
        "downsampling": [True for _ in range(4)],
        "downsampling_type": ["fused", "maxpool", "maxpool", "maxpool"],
        "reorganise": False,
        "concat_dim": 1,
        "width_multiplier": 1,
    },
    "YOLOv7-w6": {
        "initial_filters": [{"filters": 64, "kernel_size": 3, "strides": 1}],
        "filters": [64, 128, 256, 384, 512],
        "filters2": [128, 256, 512, 768, 1024],
        "num_blocks": 5,
        "depth": 2,
        "block_depth": 2,
        "downsampling": [True for _ in range(5)],
        "downsampling_type": ["fused" for _ in range(5)],
        "reorganise": True,
        "concat_dim": 1,
        "width_multiplier": 1,
    },
    "YOLOv7": {
        "initial_filters": [
            {"filters": 32, "kernel_size": 3, "strides": 1},
            {"filters": 64, "kernel_size": 3, "strides": 2},
            {"filters": 64, "kernel_size": 3, "strides": 1},
        ],
        "filters": [64, 128, 256, 256],
        "filters2": [256, 512, 1024, 1024],
        "num_blocks": 4,
        "depth": 2,
        "block_depth": 2,
        "downsampling": [True for _ in range(4)],
        "downsampling_type": [
            "fused",
            "breakdown_maxpool",
            "breakdown_maxpool",
            "breakdown_maxpool",
        ],
        "reorganise": False,
        "concat_dim": 1,
        "width_multiplier": 1,
    },
    "YOLOv7x": {
        "initial_filters": [
            {"filters": 40, "kernel_size": 3, "strides": 1},
            {"filters": 80, "kernel_size": 3, "strides": 2},
            {"filters": 80, "kernel_size": 3, "strides": 1},
        ],
        "filters": [64, 128, 256, 256],
        "filters2": [320, 640, 1280, 1280],
        "num_blocks": 4,
        "depth": 3,
        "block_depth": 2,
        "downsampling": [True for _ in range(4)],
        "downsampling_type": [
            "fused",
            "breakdown_maxpool",
            "breakdown_maxpool",
            "breakdown_maxpool",
        ],
        "reorganise": False,
        "concat_dim": 1,
        "width_multiplier": 1,
    },
}


class YOLOV7BackBone(tf.keras.layers.Layer):
    def __init__(
        self,
        initial_filters,
        filters,
        filters2,
        num_blocks,
        depth,
        block_depth,
        downsampling,
        downsampling_type,
        reorganise=True,
        concat_dim=-1,
        width_multiplier=1,
    ):
        super(YOLOV7BackBone, self).__init__()
        self.initial_filters = initial_filters
        self.filters = filters
        self.filters2 = filters2
        self.num_blocks = num_blocks
        self.depth = depth
        self.block_depth = block_depth
        self.downsampling = downsampling
        self.downsampling_type = downsampling_type
        self.width_multiplier = width_multiplier
        self.reorganise = reorganise
        self.concat_dim = concat_dim
        self.b = [
            [
                Block(depth, block_depth, filter, filter2, concat_dim)
                for _ in range(width_multiplier)
            ]
            for (filter, filter2) in zip(filters, filters2)
        ]

    def call(self, x):
        out = x
        if self.reorganise:
            out = ReOrganise()(out)
        for filter_cfg in self.initial_filters:
            out = FusedConvolution(
                filter_cfg["filters"],
                kernel_size=filter_cfg["kernel_size"],
                strides=filter_cfg["strides"],
            )(out)
        for index, i in enumerate(self.b):
            if self.downsampling[index]:
                if self.downsampling_type[index] == "maxpool":
                    out = tf.keras.layers.MaxPooling2D()(out)
                elif self.downsampling_type[index] == "fused":
                    out = FusedConvolution(
                        self.filters2[index], kernel_size=3, strides=2
                    )(out)
                elif self.downsampling_type[index] == "breakdown_maxpool":
                    out = BreakDownFusedConv(depth=2, concat_dim=self.concat_dim)(out)
                elif self.downsampling_type[index] == "downc":
                    channels = out.shape[-1]
                    out = DownC(channels * 2)(out)

            if len(i) == 1:
                out = i[0](out)
            else:
                inter = [layer(out) for layer in i]
                out = Shortcut()(inter)
        return out

    def get_config(self):
        config = super(YOLOV7BackBone, self).get_config()
        config.update(
            {
                "initial_filters": self.initial_filters,
                "filters": self.filters,
                "filters2": self.filters2,
                "num_blocks": self.num_blocks,
                "depth": self.depth,
                "block_depth": self.block_depth,
                "downsampling": self.downsampling,
                "downsampling_type": self.downsampling_type,
                "reorganise": self.reorganise,
                "concat_dim": self.concat_dim,
                "width_multiplier": self.width_multiplier,
            }
        )
        return config


def YOLOv7_d6_Backbone():
    return YOLOV7BackBone(
        initial_filters=BACKBONE_MODEL_CONFIGS["YOLOv7-d6"]["initial_filters"],
        filters=BACKBONE_MODEL_CONFIGS["YOLOv7-d6"]["filters"],
        filters2=BACKBONE_MODEL_CONFIGS["YOLOv7-d6"]["filters2"],
        num_blocks=BACKBONE_MODEL_CONFIGS["YOLOv7-d6"]["num_blocks"],
        depth=BACKBONE_MODEL_CONFIGS["YOLOv7-d6"]["depth"],
        block_depth=BACKBONE_MODEL_CONFIGS["YOLOv7-d6"]["block_depth"],
        downsampling=BACKBONE_MODEL_CONFIGS["YOLOv7-d6"]["downsampling"],
        downsampling_type=BACKBONE_MODEL_CONFIGS["YOLOv7-d6"]["downsampling_type"],
        reorganise=BACKBONE_MODEL_CONFIGS["YOLOv7-d6"]["reorganise"],
        concat_dim=BACKBONE_MODEL_CONFIGS["YOLOv7-d6"]["concat_dim"],
        width_multiplier=BACKBONE_MODEL_CONFIGS["YOLOv7-d6"]["width_multiplier"],
    )


def YOLOv7_e6_Backbone():
    return YOLOV7BackBone(
        initial_filters=BACKBONE_MODEL_CONFIGS["YOLOv7-e6"]["initial_filters"],
        filters=BACKBONE_MODEL_CONFIGS["YOLOv7-e6"]["filters"],
        filters2=BACKBONE_MODEL_CONFIGS["YOLOv7-e6"]["filters2"],
        num_blocks=BACKBONE_MODEL_CONFIGS["YOLOv7-e6"]["num_blocks"],
        depth=BACKBONE_MODEL_CONFIGS["YOLOv7-e6"]["depth"],
        block_depth=BACKBONE_MODEL_CONFIGS["YOLOv7-e6"]["block_depth"],
        downsampling=BACKBONE_MODEL_CONFIGS["YOLOv7-e6"]["downsampling"],
        downsampling_type=BACKBONE_MODEL_CONFIGS["YOLOv7-e6"]["downsampling_type"],
        reorganise=BACKBONE_MODEL_CONFIGS["YOLOv7-e6"]["reorganise"],
        concat_dim=BACKBONE_MODEL_CONFIGS["YOLOv7-e6"]["concat_dim"],
        width_multiplier=BACKBONE_MODEL_CONFIGS["YOLOv7-e6"]["width_multiplier"],
    )


def YOLOv7_e6e_Backbone():
    return YOLOV7BackBone(
        initial_filters=BACKBONE_MODEL_CONFIGS["YOLOv7-e6e"]["initial_filters"],
        filters=BACKBONE_MODEL_CONFIGS["YOLOv7-e6e"]["filters"],
        filters2=BACKBONE_MODEL_CONFIGS["YOLOv7-e6e"]["filters2"],
        num_blocks=BACKBONE_MODEL_CONFIGS["YOLOv7-e6e"]["num_blocks"],
        depth=BACKBONE_MODEL_CONFIGS["YOLOv7-e6e"]["depth"],
        block_depth=BACKBONE_MODEL_CONFIGS["YOLOv7-e6e"]["block_depth"],
        downsampling=BACKBONE_MODEL_CONFIGS["YOLOv7-e6e"]["downsampling"],
        downsampling_type=BACKBONE_MODEL_CONFIGS["YOLOv7-e6e"]["downsampling_type"],
        reorganise=BACKBONE_MODEL_CONFIGS["YOLOv7-e6e"]["reorganise"],
        concat_dim=BACKBONE_MODEL_CONFIGS["YOLOv7-e6e"]["concat_dim"],
        width_multiplier=BACKBONE_MODEL_CONFIGS["YOLOv7-e6e"]["width_multiplier"],
    )


def YOLOv7_tiny_Backbone():
    return YOLOV7BackBone(
        initial_filters=BACKBONE_MODEL_CONFIGS["YOLOv7-tiny"]["initial_filters"],
        filters=BACKBONE_MODEL_CONFIGS["YOLOv7-tiny"]["filters"],
        filters2=BACKBONE_MODEL_CONFIGS["YOLOv7-tiny"]["filters2"],
        num_blocks=BACKBONE_MODEL_CONFIGS["YOLOv7-tiny"]["num_blocks"],
        depth=BACKBONE_MODEL_CONFIGS["YOLOv7-tiny"]["depth"],
        block_depth=BACKBONE_MODEL_CONFIGS["YOLOv7-tiny"]["block_depth"],
        downsampling=BACKBONE_MODEL_CONFIGS["YOLOv7-tiny"]["downsampling"],
        downsampling_type=BACKBONE_MODEL_CONFIGS["YOLOv7-tiny"]["downsampling_type"],
        reorganise=BACKBONE_MODEL_CONFIGS["YOLOv7-tiny"]["reorganise"],
        concat_dim=BACKBONE_MODEL_CONFIGS["YOLOv7-tiny"]["concat_dim"],
        width_multiplier=BACKBONE_MODEL_CONFIGS["YOLOv7-tiny"]["width_multiplier"],
    )


def YOLOv7_w6_Backbone():
    return YOLOV7BackBone(
        initial_filters=BACKBONE_MODEL_CONFIGS["YOLOv7-w6"]["initial_filters"],
        filters=BACKBONE_MODEL_CONFIGS["YOLOv7-w6"]["filters"],
        filters2=BACKBONE_MODEL_CONFIGS["YOLOv7-w6"]["filters2"],
        num_blocks=BACKBONE_MODEL_CONFIGS["YOLOv7-w6"]["num_blocks"],
        depth=BACKBONE_MODEL_CONFIGS["YOLOv7-w6"]["depth"],
        block_depth=BACKBONE_MODEL_CONFIGS["YOLOv7-w6"]["block_depth"],
        downsampling=BACKBONE_MODEL_CONFIGS["YOLOv7-w6"]["downsampling"],
        downsampling_type=BACKBONE_MODEL_CONFIGS["YOLOv7-w6"]["downsampling_type"],
        reorganise=BACKBONE_MODEL_CONFIGS["YOLOv7-w6"]["reorganise"],
        concat_dim=BACKBONE_MODEL_CONFIGS["YOLOv7-w6"]["concat_dim"],
        width_multiplier=BACKBONE_MODEL_CONFIGS["YOLOv7-w6"]["width_multiplier"],
    )


def YOLOv7_Backbone():
    return YOLOV7BackBone(
        initial_filters=BACKBONE_MODEL_CONFIGS["YOLOv7"]["initial_filters"],
        filters=BACKBONE_MODEL_CONFIGS["YOLOv7"]["filters"],
        filters2=BACKBONE_MODEL_CONFIGS["YOLOv7"]["filters2"],
        num_blocks=BACKBONE_MODEL_CONFIGS["YOLOv7"]["num_blocks"],
        depth=BACKBONE_MODEL_CONFIGS["YOLOv7"]["depth"],
        block_depth=BACKBONE_MODEL_CONFIGS["YOLOv7"]["block_depth"],
        downsampling=BACKBONE_MODEL_CONFIGS["YOLOv7"]["downsampling"],
        downsampling_type=BACKBONE_MODEL_CONFIGS["YOLOv7"]["downsampling_type"],
        reorganise=BACKBONE_MODEL_CONFIGS["YOLOv7"]["reorganise"],
        concat_dim=BACKBONE_MODEL_CONFIGS["YOLOv7"]["concat_dim"],
        width_multiplier=BACKBONE_MODEL_CONFIGS["YOLOv7"]["width_multiplier"],
    )


def YOLOv7x_Backbone():
    return YOLOV7BackBone(
        initial_filters=BACKBONE_MODEL_CONFIGS["YOLOv7x"]["initial_filters"],
        filters=BACKBONE_MODEL_CONFIGS["YOLOv7x"]["filters"],
        filters2=BACKBONE_MODEL_CONFIGS["YOLOv7x"]["filters2"],
        num_blocks=BACKBONE_MODEL_CONFIGS["YOLOv7x"]["num_blocks"],
        depth=BACKBONE_MODEL_CONFIGS["YOLOv7x"]["depth"],
        block_depth=BACKBONE_MODEL_CONFIGS["YOLOv7x"]["block_depth"],
        downsampling=BACKBONE_MODEL_CONFIGS["YOLOv7x"]["downsampling"],
        downsampling_type=BACKBONE_MODEL_CONFIGS["YOLOv7x"]["downsampling_type"],
        reorganise=BACKBONE_MODEL_CONFIGS["YOLOv7x"]["reorganise"],
        concat_dim=BACKBONE_MODEL_CONFIGS["YOLOv7x"]["concat_dim"],
        width_multiplier=BACKBONE_MODEL_CONFIGS["YOLOv7x"]["width_multiplier"],
    )
