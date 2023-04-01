# Copyright 2023 The KerasCV Authors
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

"""MobileNetV3 model preset configurations."""

from tensorflow.keras import layers

import keras_cv.models.backbones.mobilenet_v3.mobilenet_v3_backbone as backbone


def stack_fn_s(x, kernel, activation, se_ratio, alpha=1.0):
    x = backbone.apply_inverted_res_block(
        x, 1, backbone.depth(16 * alpha), 3, 2, se_ratio, layers.ReLU(), 0
    )
    x = backbone.apply_inverted_res_block(
        x, 72.0 / 16, backbone.depth(24 * alpha), 3, 2, None, layers.ReLU(), 1
    )
    x = backbone.apply_inverted_res_block(
        x, 88.0 / 24, backbone.depth(24 * alpha), 3, 1, None, layers.ReLU(), 2
    )
    x = backbone.apply_inverted_res_block(
        x, 4, backbone.depth(40 * alpha), kernel, 2, se_ratio, activation, 3
    )
    x = backbone.apply_inverted_res_block(
        x, 6, backbone.depth(40 * alpha), kernel, 1, se_ratio, activation, 4
    )
    x = backbone.apply_inverted_res_block(
        x, 6, backbone.depth(40 * alpha), kernel, 1, se_ratio, activation, 5
    )
    x = backbone.apply_inverted_res_block(
        x, 3, backbone.depth(48 * alpha), kernel, 1, se_ratio, activation, 6
    )
    x = backbone.apply_inverted_res_block(
        x, 3, backbone.depth(48 * alpha), kernel, 1, se_ratio, activation, 7
    )
    x = backbone.apply_inverted_res_block(
        x, 6, backbone.depth(96 * alpha), kernel, 2, se_ratio, activation, 8
    )
    x = backbone.apply_inverted_res_block(
        x, 6, backbone.depth(96 * alpha), kernel, 1, se_ratio, activation, 9
    )
    x = backbone.apply_inverted_res_block(
        x, 6, backbone.depth(96 * alpha), kernel, 1, se_ratio, activation, 10
    )
    return x


def stack_fn_l(x, kernel, activation, se_ratio, alpha=1.0):
    x = backbone.apply_inverted_res_block(
        x, 1, backbone.depth(16 * alpha), 3, 1, None, layers.ReLU(), 0
    )
    x = backbone.apply_inverted_res_block(
        x, 4, backbone.depth(24 * alpha), 3, 2, None, layers.ReLU(), 1
    )
    x = backbone.apply_inverted_res_block(
        x, 3, backbone.depth(24 * alpha), 3, 1, None, layers.ReLU(), 2
    )
    x = backbone.apply_inverted_res_block(
        x, 3, backbone.depth(40 * alpha), kernel, 2, se_ratio, layers.ReLU(), 3
    )
    x = backbone.apply_inverted_res_block(
        x, 3, backbone.depth(40 * alpha), kernel, 1, se_ratio, layers.ReLU(), 4
    )
    x = backbone.apply_inverted_res_block(
        x, 3, backbone.depth(40 * alpha), kernel, 1, se_ratio, layers.ReLU(), 5
    )
    x = backbone.apply_inverted_res_block(
        x, 6, backbone.depth(80 * alpha), 3, 2, None, activation, 6
    )
    x = backbone.apply_inverted_res_block(
        x, 2.5, backbone.depth(80 * alpha), 3, 1, None, activation, 7
    )
    x = backbone.apply_inverted_res_block(
        x, 2.3, backbone.depth(80 * alpha), 3, 1, None, activation, 8
    )
    x = backbone.apply_inverted_res_block(
        x, 2.3, backbone.depth(80 * alpha), 3, 1, None, activation, 9
    )
    x = backbone.apply_inverted_res_block(
        x, 6, backbone.depth(112 * alpha), 3, 1, se_ratio, activation, 10
    )
    x = backbone.apply_inverted_res_block(
        x, 6, backbone.depth(112 * alpha), 3, 1, se_ratio, activation, 11
    )
    x = backbone.apply_inverted_res_block(
        x, 6, backbone.depth(160 * alpha), kernel, 2, se_ratio, activation, 12
    )
    x = backbone.apply_inverted_res_block(
        x, 6, backbone.depth(160 * alpha), kernel, 1, se_ratio, activation, 13
    )
    x = backbone.apply_inverted_res_block(
        x, 6, backbone.depth(160 * alpha), kernel, 1, se_ratio, activation, 14
    )
    return x


backbone_presets_no_weights = {
    "mobilenetv3small": {
        "metadata": {
            "description": (
                "MobileNetV3 model with 14 layers where the batch "
                "normalization and hard-swish activation are applied after the "
                "convolution layers."
            ),
        },
        "class_name": "keras_cv.models>MobileNetV3Backbone",
        "config": {
            "stack_fn": stack_fn_s,
            "last_point_ch": 1024,
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "alpha": 1.0,
            "minimalistic": True,
            "dropout_rate": 0.2,
        },
    },
    "mobilenetv3large": {
        "metadata": {
            "description": (
                "MobileNetV3 model with 28 layers where the batch "
                "normalization and hard-swish activation are applied after the "
                "convolution layers."
            ),
        },
        "class_name": "keras_cv.models>MobileNetV3Backbone",
        "config": {
            "stack_fn": stack_fn_l,
            "last_point_ch": 1280,
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "alpha": 1.0,
            "minimalistic": True,
            "dropout_rate": 0.2,
        },
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
}
