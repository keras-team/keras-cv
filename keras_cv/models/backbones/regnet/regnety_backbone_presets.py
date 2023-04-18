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
    "regnety002": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [1, 1, 4, 7],
            "widths": [24, 56, 152, 368],
            "group_width": 8,
            "default_size": 224,
            "block_type": "Y",
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
    "regnety004": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [1, 3, 6, 6],
            "widths": [48, 104, 208, 440],
            "group_width": 8,
            "default_size": 224,
            "block_type": "Y",
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
    "regnety006": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [1, 3, 7, 4],
            "widths": [48, 112, 256, 608],
            "group_width": 16,
            "default_size": 224,
            "block_type": "Y",
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
    "regnety008": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [1, 3, 8, 2],
            "widths": [64, 128, 320, 768],
            "group_width": 16,
            "default_size": 224,
            "block_type": "Y",
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
    "regnety016": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 6, 17, 2],
            "widths": [48, 120, 336, 888],
            "group_width": 24,
            "default_size": 224,
            "block_type": "Y",
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
    "regnety032": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 5, 13, 1],
            "widths": [72, 216, 576, 1512],
            "group_width": 24,
            "default_size": 224,
            "block_type": "Y",
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
    "regnety040": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 6, 12, 2],
            "widths": [128, 192, 512, 1088],
            "group_width": 64,
            "default_size": 224,
            "block_type": "Y",
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
    "regnety064": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 7, 14, 2],
            "widths": [144, 288, 576, 1296],
            "group_width": 72,
            "default_size": 224,
            "block_type": "Y",
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
    "regnety080": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 4, 10, 1],
            "widths": [168, 448, 896, 2016],
            "group_width": 56,
            "default_size": 224,
            "block_type": "Y",
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
    "regnety120": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 5, 11, 1],
            "widths": [224, 448, 896, 2240],
            "group_width": 112,
            "default_size": 224,
            "block_type": "Y",
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
    "regnety160": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 4, 11, 1],
            "widths": [224, 448, 1232, 3024],
            "group_width": 112,
            "default_size": 224,
            "block_type": "Y",
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
    "regnety320": {
        "metadata": {
            "description": ("RegNet Model"),
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 5, 12, 1],
            "widths": [232, 696, 1392, 3712],
            "group_width": 232,
            "default_size": 224,
            "block_type": "Y",
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

backbone_presets_with_weights = {}

backbone_presets_y = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
