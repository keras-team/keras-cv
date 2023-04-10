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
"""RegNet model preset configurations."""
# The widths and depths are deduced from a quantized linear function. For
# more information, please refer to "Designing Network Design Spaces" by
# Radosavovic et al.

backbone_presets_no_weights = {
    "regnetx002": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [1, 1, 4, 7],
            "widths": [24, 56, 152, 368],
            "group_width": 8,
            "default_size": 224,
            "block_type": "X",
            "include_rescaling": True,
            "include_top": False,
            "num_classes": None,
            "weights": None,
            "input_tensor": None,
            "input_shape": (None, None, 3),
            "pooling": None,
            "classifier_activation": "softmax",
        },
    },
    "regnetx004": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [1, 2, 7, 12],
            "widths": [32, 64, 160, 384],
            "group_width": 16,
            "default_size": 224,
            "block_type": "X",
            "include_rescaling": True,
            "include_top": False,
            "num_classes": None,
            "weights": None,
            "input_tensor": None,
            "input_shape": (None, None, 3),
            "pooling": None,
            "classifier_activation": "softmax",
        },
    },
    "regnetx006": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [1, 3, 5, 7],
            "widths": [48, 96, 240, 528],
            "group_width": 24,
            "default_size": 224,
            "block_type": "X",
            "include_rescaling": True,
            "include_top": False,
            "num_classes": None,
            "weights": None,
            "input_tensor": None,
            "input_shape": (None, None, 3),
            "pooling": None,
            "classifier_activation": "softmax",
        },
    },
    "regnetx008": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [1, 3, 7, 5],
            "widths": [64, 128, 288, 672],
            "group_width": 16,
            "default_size": 224,
            "block_type": "X",
            "include_rescaling": True,
            "include_top": False,
            "num_classes": None,
            "weights": None,
            "input_tensor": None,
            "input_shape": (None, None, 3),
            "pooling": None,
            "classifier_activation": "softmax",
        },
    },
    "regnetx016": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 4, 10, 2],
            "widths": [72, 168, 408, 912],
            "group_width": 24,
            "default_size": 224,
            "block_type": "X",
            "include_rescaling": True,
            "include_top": False,
            "num_classes": None,
            "weights": None,
            "input_tensor": None,
            "input_shape": (None, None, 3),
            "pooling": None,
            "classifier_activation": "softmax",
        },
    },
    "regnetx032": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 6, 15, 2],
            "widths": [96, 192, 432, 1008],
            "group_width": 48,
            "default_size": 224,
            "block_type": "X",
            "include_rescaling": True,
            "include_top": False,
            "num_classes": None,
            "weights": None,
            "input_tensor": None,
            "input_shape": (None, None, 3),
            "pooling": None,
            "classifier_activation": "softmax",
        },
    },
    "regnetx040": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 5, 14, 2],
            "widths": [80, 240, 560, 1360],
            "group_width": 40,
            "default_size": 224,
            "block_type": "X",
            "include_rescaling": True,
            "include_top": False,
            "num_classes": None,
            "weights": None,
            "input_tensor": None,
            "input_shape": (None, None, 3),
            "pooling": None,
            "classifier_activation": "softmax",
        },
    },
    "regnetx064": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 4, 10, 1],
            "widths": [168, 392, 784, 1624],
            "group_width": 56,
            "default_size": 224,
            "block_type": "X",
            "include_rescaling": True,
            "include_top": False,
            "num_classes": None,
            "weights": None,
            "input_tensor": None,
            "input_shape": (None, None, 3),
            "pooling": None,
            "classifier_activation": "softmax",
        },
    },
    "regnetx080": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 5, 15, 1],
            "widths": [80, 240, 720, 1920],
            "group_width": 120,
            "default_size": 224,
            "block_type": "X",
            "include_rescaling": True,
            "include_top": False,
            "num_classes": None,
            "weights": None,
            "input_tensor": None,
            "input_shape": (None, None, 3),
            "pooling": None,
            "classifier_activation": "softmax",
        },
    },
    "regnetx120": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 5, 11, 1],
            "widths": [224, 448, 896, 2240],
            "group_width": 112,
            "default_size": 224,
            "block_type": "X",
            "include_rescaling": True,
            "include_top": False,
            "num_classes": None,
            "weights": None,
            "input_tensor": None,
            "input_shape": (None, None, 3),
            "pooling": None,
            "classifier_activation": "softmax",
        },
    },
    "regnetx160": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 6, 13, 1],
            "widths": [256, 512, 896, 2048],
            "group_width": 128,
            "default_size": 224,
            "block_type": "X",
            "include_rescaling": True,
            "include_top": False,
            "num_classes": None,
            "weights": None,
            "input_tensor": None,
            "input_shape": (None, None, 3),
            "pooling": None,
            "classifier_activation": "softmax",
        },
    },
    "regnetx320": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 7, 13, 1],
            "widths": [336, 672, 1344, 2520],
            "group_width": 168,
            "default_size": 224,
            "block_type": "X",
            "include_rescaling": True,
            "include_top": False,
            "num_classes": None,
            "weights": None,
            "input_tensor": None,
            "input_shape": (None, None, 3),
            "pooling": None,
            "classifier_activation": "softmax",
        },
    },
}

backbone_presets_x = {
    **backbone_presets_no_weights,
}
