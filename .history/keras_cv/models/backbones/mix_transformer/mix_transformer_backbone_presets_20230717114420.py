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
"""MiT model preset configurations."""

backbone_presets_no_weights = {
    "MiT_B0": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 8 transformer blocks."
            ),
            "params": 3321962,
            "official_name": "MiT",
            "path": "mit",
        },
        "class_name": "keras_cv.models>MobileNetV3Backbone",
        "config": {
            "embedding_dims": [32, 64, 160, 256],
            "depths": [2, 2, 2, 2],
        },
    },
    "MiT_B1": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 8 transformer blocks."
            ),
            "params": 13156554,
            "official_name": "MiT",
            "path": "mit",
        },
        "class_name": "keras_cv.models>MobileNetV3Backbone",
        "config": {
            "embedding_dims": [64, 128, 320, 512],
            "depths": [2, 2, 2, 2],
        },
    },
    "MiT_B2": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 16 transformer blocks."
            ),
            "params": 24201418,
            "official_name": "MiT",
            "path": "mit",
        },
        "class_name": "keras_cv.models>MobileNetV3Backbone",
        "config": {
            "embedding_dims": [64, 128, 320, 512],
            "depths": [3, 4, 6, 3],
        },
    },
    "MiT_B3": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 28 transformer blocks."
            ),
            "params": 44077258,
            "official_name": "MiT",
            "path": "mit",
        },
        "class_name": "keras_cv.models>MobileNetV3Backbone",
        "config": {
            "embedding_dims": [64, 128, 320, 512],
            "depths": [3, 4, 18, 3],
        },
    },
    "MiT_B4": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 41 transformer blocks."
            ),
            "params": 60847818,
            "official_name": "MiT",
            "path": "mit",
        },
        "class_name": "keras_cv.models>MobileNetV3Backbone",
        "config": {
            "embedding_dims": [64, 128, 320, 512],
            "depths": [3, 8, 27, 3],
        },
    },
    "MiT_B5": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 52 transformer blocks."
            ),
            "params": 81448138,
            "official_name": "MiT",
            "path": "mit",
        },
        "class_name": "keras_cv.models>MobileNetV3Backbone",
        "config": {
            "embedding_dims": [64, 128, 320, 512],
            "depths": [3, 6, 40, 3],
        },
    },
}

backbone_presets_with_weights = {}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
