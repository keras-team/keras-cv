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

"""EfficientNetV2 model preset configurations."""

backbone_presets_no_weights = {
    "efficientnetv2-s": {
        "metadata": {
            "description": ("EfficientNetv2 model"),
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "default_size": 384,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "bn_momentum": 0.9,
            "activation": "swish",
            "blocks_args": "default",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "efficientnetv2-m": {
        "metadata": {
            "description": ("EfficientNetv2 model"),
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "default_size": 480,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "bn_momentum": 0.9,
            "activation": "swish",
            "blocks_args": "default",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "efficientnetv2-l": {
        "metadata": {
            "description": ("EfficientNetv2 model"),
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "default_size": 480,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "bn_momentum": 0.9,
            "activation": "swish",
            "blocks_args": "default",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "efficientnetv2-b0": {
        "metadata": {
            "description": ("EfficientNetv2 model "),
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "default_size": 224,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "bn_momentum": 0.9,
            "activation": "swish",
            "blocks_args": "default",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "efficientnetv2-b1": {
        "metadata": {
            "description": ("EfficientNetv2 model "),
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "width_coefficient": 1.0,
            "depth_coefficient": 1.1,
            "default_size": 240,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "bn_momentum": 0.9,
            "activation": "swish",
            "blocks_args": "default",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "efficientnetv2-b2": {
        "metadata": {
            "description": ("EfficientNetv2 model "),
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "width_coefficient": 1.1,
            "depth_coefficient": 1.2,
            "default_size": 260,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "bn_momentum": 0.9,
            "activation": "swish",
            "blocks_args": "default",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "efficientnetv2-b3": {
        "metadata": {
            "description": ("EfficientNetv2 model "),
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "width_coefficient": 1.2,
            "depth_coefficient": 1.4,
            "default_size": 300,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "bn_momentum": 0.9,
            "activation": "swish",
            "blocks_args": "default",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
}

backbone_presets_with_weights = {}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
