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
            "description": ("Regnet Model"),
            "params": 2336640,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [1, 1, 4, 7],
            "widths": [24, 56, 152, 368],
            "group_width": 8,
            "block_type": "X",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
    "regnetx004": {
        "metadata": {
            "description": ("Regnet Model"),
            "params": 4809184,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [1, 2, 7, 12],
            "widths": [32, 64, 160, 384],
            "group_width": 16,
            "block_type": "X",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
    "regnetx006": {
        "metadata": {
            "description": ("Regnet Model"),
            "params": 5700320,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [1, 3, 5, 7],
            "widths": [48, 96, 240, 528],
            "group_width": 24,
            "block_type": "X",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
    "regnetx008": {
        "metadata": {
            "description": ("Regnet Model"),
            "params": 6623968,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [1, 3, 7, 5],
            "widths": [64, 128, 288, 672],
            "group_width": 16,
            "block_type": "X",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
    "regnetx016": {
        "metadata": {
            "description": ("Regnet Model"),
            "params": 8320640,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 4, 10, 2],
            "widths": [72, 168, 408, 912],
            "group_width": 24,
            "block_type": "X",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
    "regnetx032": {
        "metadata": {
            "description": ("Regnet Model"),
            "params": 14350112,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 6, 15, 2],
            "widths": [96, 192, 432, 1008],
            "group_width": 48,
            "block_type": "X",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
    "regnetx040": {
        "metadata": {
            "description": ("Regnet Model"),
            "params": 20833312,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 5, 14, 2],
            "widths": [80, 240, 560, 1360],
            "group_width": 40,
            "block_type": "X",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
    "regnetx064": {
        "metadata": {
            "description": ("Regnet Model"),
            "params": 24658464,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 4, 10, 1],
            "widths": [168, 392, 784, 1624],
            "group_width": 56,
            "block_type": "X",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
    "regnetx080": {
        "metadata": {
            "description": ("Regnet Model"),
            "params": 37742112,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 5, 15, 1],
            "widths": [80, 240, 720, 1920],
            "group_width": 120,
            "block_type": "X",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
    "regnetx120": {
        "metadata": {
            "description": ("Regnet Model"),
            "params": 43961440,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 5, 11, 1],
            "widths": [224, 448, 896, 2240],
            "group_width": 112,
            "block_type": "X",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
    "regnetx160": {
        "metadata": {
            "description": ("Regnet Model"),
            "params": 52340704,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 6, 13, 1],
            "widths": [256, 512, 896, 2048],
            "group_width": 128,
            "block_type": "X",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
    "regnetx320": {
        "metadata": {
            "description": ("Regnet Model"),
            "params": 105452576,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 7, 13, 1],
            "widths": [336, 672, 1344, 2520],
            "group_width": 168,
            "block_type": "X",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
    "regnety002": {
        "metadata": {
            "description": ("Regnet Model"),
            "params": 2814844,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [1, 1, 4, 7],
            "widths": [24, 56, 152, 368],
            "group_width": 8,
            "block_type": "Y",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
    "regnety004": {
        "metadata": {
            "description": ("Regnet Model"),
            "params": 3930296,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [1, 3, 6, 6],
            "widths": [48, 104, 208, 440],
            "group_width": 8,
            "block_type": "Y",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
    "regnety006": {
        "metadata": {
            "description": ("Regnet Model"),
            "params": 5475920,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [1, 3, 7, 4],
            "widths": [48, 112, 256, 608],
            "group_width": 16,
            "block_type": "Y",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
    "regnety008": {
        "metadata": {
            "description": ("Regnet Model"),
            "params": 5524056,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [1, 3, 8, 2],
            "widths": [64, 128, 320, 768],
            "group_width": 16,
            "block_type": "Y",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
    "regnety016": {
        "metadata": {
            "description": ("Regnet Model"),
            "params": 10366102,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 6, 17, 2],
            "widths": [48, 120, 336, 888],
            "group_width": 24,
            "block_type": "Y",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
    "regnety032": {
        "metadata": {
            "description": ("Regnet Model"),
            "params": 17989498,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 5, 13, 1],
            "widths": [72, 216, 576, 1512],
            "group_width": 24,
            "block_type": "Y",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
    "regnety040": {
        "metadata": {
            "description": ("Regnet Model"),
            "params": 19619928,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 6, 12, 2],
            "widths": [128, 192, 512, 1088],
            "group_width": 64,
            "block_type": "Y",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
    "regnety064": {
        "metadata": {
            "description": ("Regnet Model"),
            "params": 29368684,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 7, 14, 2],
            "widths": [144, 288, 576, 1296],
            "group_width": 72,
            "block_type": "Y",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
    "regnety080": {
        "metadata": {
            "description": ("Regnet Model"),
            "params": 37248812,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 4, 10, 1],
            "widths": [168, 448, 896, 2016],
            "group_width": 56,
            "block_type": "Y",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
    "regnety120": {
        "metadata": {
            "description": ("Regnet Model"),
            "params": 49677928,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 5, 11, 1],
            "widths": [224, 448, 896, 2240],
            "group_width": 112,
            "block_type": "Y",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
    "regnety160": {
        "metadata": {
            "description": ("Regnet Model"),
            "params": 80687956,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 4, 11, 1],
            "widths": [224, 448, 1232, 3024],
            "group_width": 112,
            "block_type": "Y",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
    "regnety320": {
        "metadata": {
            "description": ("Regnet Model"),
            "params": 141492058,
            "official_name": "Regnet",
            "path": "Regnet",
        },
        "class_name": "keras_cv.models>RegNetBackbone",
        "config": {
            "depths": [2, 5, 12, 1],
            "widths": [232, 696, 1392, 3712],
            "group_width": 232,
            "block_type": "Y",
            "include_rescaling": True,
            "input_tensor": None,
            "input_shape": (None, None, 3),
        },
    },
}

backbone_presets_with_weights = {}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
