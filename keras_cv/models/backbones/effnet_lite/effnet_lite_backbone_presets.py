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
"""EfficientNetLite model preset configurations."""

backbone_presets_no_weights = {
    "efficientnetliteb0": {
        "metadata": {
            "description": (
                "EfficientNetLite model with 1.0 depth coefficient and 1.0 width "
                "coefficient where the batch normalization and ReLU6 as default activation "
                "are applied after the convolution layers."
            ),
        },
        "class_name": "keras_cv.models>EfficientNetLiteBackbone",
        "config": {
            "include_rescaling": True,
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            # "default_size": default_size,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "activation": relu6,
            "blocks_args": None,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "efficientnetliteb1": {
        "metadata": {
            "description": (
                "EfficientNetLite model with 1.0 depth coefficient and 1.1 width "
                "coefficient where the batch normalization and ReLU6 as default activation "
                "are applied after the convolution layers."
            ),
        },
        "class_name": "keras_cv.models>EfficientNetLiteBackbone",
        "config": {
            "include_rescaling": True,
            "width_coefficient": 1.0,
            "depth_coefficient": 1.1,
            # "default_size": default_size,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "activation": relu6,
            "blocks_args": None,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "efficientnetliteb2": {
        "metadata": {
            "description": (
                "EfficientNetLite model with 1.1 depth coefficient and 1.2 width "
                "coefficient where the batch normalization and ReLU6 as default activation "
                "are applied after the convolution layers."
            ),
        },
        "class_name": "keras_cv.models>EfficientNetLiteBackbone",
        "config": {
            "include_rescaling": True,
            "width_coefficient": 1.1,
            "depth_coefficient": 1.2,
            # "default_size": default_size,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "activation": relu6,
            "blocks_args": None,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "efficientnetliteb3": {
        "metadata": {
            "description": (
                "EfficientNetLite model with 1.2 depth coefficient and 1.4 width "
                "coefficient where the batch normalization and ReLU6 as default activation "
                "are applied after the convolution layers."
            ),
        },
        "class_name": "keras_cv.models>EfficientNetLiteBackbone",
        "config": {
            "include_rescaling": True,
            "width_coefficient": 1.2,
            "depth_coefficient": 1.4,
            # "default_size": default_size,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "activation": relu6,
            "blocks_args": None,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "efficientnetliteb4": {
        "metadata": {
            "description": (
                "EfficientNetLite model with 1.4 depth coefficient and 1.8 width "
                "coefficient where the batch normalization and ReLU6 as default activation "
                "are applied after the convolution layers."
            ),
        },
        "class_name": "keras_cv.models>EfficientNetLiteBackbone",
        "config": {
            "include_rescaling": True,
            "width_coefficient": 1.4,
            "depth_coefficient": 1.8,
            # "default_size": default_size,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "activation": relu6,
            "blocks_args": None,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
}


backbone_presets = {
    **backbone_presets_no_weights,
}
