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

from keras_cv.models.backbones.mobilenet_v3 import (
    mobilenet_v3_backbone as backbone,
)

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
            "stackwise_expansion": [
                1,
                72.0 / 16,
                88.0 / 24,
                4,
                6,
                6,
                3,
                3,
                6,
                6,
                6,
            ],
            "stackwise_filters": [16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96],
            "stackwise_kernel_size": [3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5],
            "stackwise_stride": [2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1],
            "stackwise_se_ratio": [
                0.25,
                None,
                None,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
            ],
            "stackwise_activation": [
                layers.ReLU(),
                layers.ReLU(),
                layers.ReLU(),
                backbone.apply_hard_swish,
                backbone.apply_hard_swish,
                backbone.apply_hard_swish,
                backbone.apply_hard_swish,
                backbone.apply_hard_swish,
                backbone.apply_hard_swish,
                backbone.apply_hard_swish,
                backbone.apply_hard_swish,
            ],
            "filters": 1024,
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "alpha": 1.0,
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
            "stackwise_expansion": [
                1,
                4,
                3,
                3,
                3,
                3,
                6,
                2.5,
                2.3,
                2.3,
                6,
                6,
                6,
                6,
                6,
            ],
            "stackwise_filters": [
                16,
                24,
                24,
                40,
                40,
                40,
                80,
                80,
                80,
                80,
                112,
                112,
                160,
                160,
                160,
            ],
            "stackwise_kernel_size": [
                3,
                3,
                3,
                5,
                5,
                5,
                3,
                3,
                3,
                3,
                3,
                3,
                5,
                5,
                5,
            ],
            "stackwise_stride": [1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1],
            "stackwise_se_ratio": [
                None,
                None,
                None,
                0.25,
                0.25,
                0.25,
                None,
                None,
                None,
                None,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
            ],
            "stackwise_activation": [
                layers.ReLU(),
                layers.ReLU(),
                layers.ReLU(),
                layers.ReLU(),
                layers.ReLU(),
                layers.ReLU(),
                backbone.apply_hard_swish,
                backbone.apply_hard_swish,
                backbone.apply_hard_swish,
                backbone.apply_hard_swish,
                backbone.apply_hard_swish,
                backbone.apply_hard_swish,
                backbone.apply_hard_swish,
                backbone.apply_hard_swish,
                backbone.apply_hard_swish,
            ],
            "filters": 1280,
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "alpha": 1.0,
            "dropout_rate": 0.2,
        },
    },
}

backbone_presets_with_weights = {}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
