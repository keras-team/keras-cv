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

"""EfficientNetV1 model preset configurations."""

DESCRIPTION = "One of the many EfficientNetV1 variants.  Each variant is built"
"based on one of the parameterizations described in the original EfficientNetV1"
"publication.  To learn more about the parameterizations and their tradeoffs,"
"please check keras.io.  As a starting point, we recommend starting with the"
'"efficientnetv1_b0" architecture, and increasing in size to'
'"efficientnetv1_b4" or "efficientnetv1_b7" if resources permit.'

backbone_presets_no_weights = {
    "efficientnetv1_b0": {
        "metadata": {
            "description": DESCRIPTION,
        },
        "class_name": "keras_cv.models>EfficientNetV1Backbone",
        "config": {
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "stackwise_kernel_sizes": [3, 3, 5, 3, 5, 5, 3],
            "stackwise_num_repeats": [1, 2, 2, 3, 3, 4, 1],
            "stackwise_input_filters": [32, 16, 24, 40, 80, 112, 192],
            "stackwise_output_filters": [16, 24, 40, 80, 112, 192, 320],
            "stackwise_expansion_ratios": [1, 6, 6, 6, 6, 6, 6],
            "stackwise_id_skip": True,
            "stackwise_strides": [1, 2, 2, 2, 1, 2, 1],
            "stackwise_squeeze_and_excite_ratios": [
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
            ],
        },
    },
    "efficientnetv1_b1": {
        "metadata": {
            "description": DESCRIPTION,
        },
        "class_name": "keras_cv.models>EfficientNetV1Backbone",
        "config": {
            "width_coefficient": 1.0,
            "depth_coefficient": 1.1,
            "dropout_rate": 0.2,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "stackwise_kernel_sizes": [3, 3, 5, 3, 5, 5, 3],
            "stackwise_num_repeats": [1, 2, 2, 3, 3, 4, 1],
            "stackwise_input_filters": [32, 16, 24, 40, 80, 112, 192],
            "stackwise_output_filters": [16, 24, 40, 80, 112, 192, 320],
            "stackwise_expansion_ratios": [1, 6, 6, 6, 6, 6, 6],
            "stackwise_id_skip": True,
            "stackwise_strides": [1, 2, 2, 2, 1, 2, 1],
            "stackwise_squeeze_and_excite_ratios": [
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
            ],
        },
    },
    "efficientnetv1_b2": {
        "metadata": {
            "description": DESCRIPTION,
        },
        "class_name": "keras_cv.models>EfficientNetV1Backbone",
        "config": {
            "width_coefficient": 1.1,
            "depth_coefficient": 1.2,
            "dropout_rate": 0.3,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "stackwise_kernel_sizes": [3, 3, 5, 3, 5, 5, 3],
            "stackwise_num_repeats": [1, 2, 2, 3, 3, 4, 1],
            "stackwise_input_filters": [32, 16, 24, 40, 80, 112, 192],
            "stackwise_output_filters": [16, 24, 40, 80, 112, 192, 320],
            "stackwise_expansion_ratios": [1, 6, 6, 6, 6, 6, 6],
            "stackwise_id_skip": True,
            "stackwise_strides": [1, 2, 2, 2, 1, 2, 1],
            "stackwise_squeeze_and_excite_ratios": [
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
            ],
        },
    },
    "efficientnetv1_b3": {
        "metadata": {
            "description": DESCRIPTION,
        },
        "class_name": "keras_cv.models>EfficientNetV1Backbone",
        "config": {
            "width_coefficient": 1.2,
            "depth_coefficient": 1.4,
            "dropout_rate": 0.3,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "stackwise_kernel_sizes": [3, 3, 5, 3, 5, 5, 3],
            "stackwise_num_repeats": [1, 2, 2, 3, 3, 4, 1],
            "stackwise_input_filters": [32, 16, 24, 40, 80, 112, 192],
            "stackwise_output_filters": [16, 24, 40, 80, 112, 192, 320],
            "stackwise_expansion_ratios": [1, 6, 6, 6, 6, 6, 6],
            "stackwise_id_skip": True,
            "stackwise_strides": [1, 2, 2, 2, 1, 2, 1],
            "stackwise_squeeze_and_excite_ratios": [
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
            ],
        },
    },
    "efficientnetv1_b4": {
        "metadata": {
            "description": DESCRIPTION,
        },
        "class_name": "keras_cv.models>EfficientNetV1Backbone",
        "config": {
            "width_coefficient": 1.4,
            "depth_coefficient": 1.8,
            "dropout_rate": 0.4,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "stackwise_kernel_sizes": [3, 3, 5, 3, 5, 5, 3],
            "stackwise_num_repeats": [1, 2, 2, 3, 3, 4, 1],
            "stackwise_input_filters": [32, 16, 24, 40, 80, 112, 192],
            "stackwise_output_filters": [16, 24, 40, 80, 112, 192, 320],
            "stackwise_expansion_ratios": [1, 6, 6, 6, 6, 6, 6],
            "stackwise_id_skip": True,
            "stackwise_strides": [1, 2, 2, 2, 1, 2, 1],
            "stackwise_squeeze_and_excite_ratios": [
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
            ],
        },
    },
    "efficientnetv1_b5": {
        "metadata": {
            "description": DESCRIPTION,
        },
        "class_name": "keras_cv.models>EfficientNetV1Backbone",
        "config": {
            "width_coefficient": 1.6,
            "depth_coefficient": 2.2,
            "dropout_rate": 0.4,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "stackwise_kernel_sizes": [3, 3, 5, 3, 5, 5, 3],
            "stackwise_num_repeats": [1, 2, 2, 3, 3, 4, 1],
            "stackwise_input_filters": [32, 16, 24, 40, 80, 112, 192],
            "stackwise_output_filters": [16, 24, 40, 80, 112, 192, 320],
            "stackwise_expansion_ratios": [1, 6, 6, 6, 6, 6, 6],
            "stackwise_id_skip": True,
            "stackwise_strides": [1, 2, 2, 2, 1, 2, 1],
            "stackwise_squeeze_and_excite_ratios": [
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
            ],
        },
    },
    "efficientnetv1_b6": {
        "metadata": {
            "description": DESCRIPTION,
        },
        "class_name": "keras_cv.models>EfficientNetV1Backbone",
        "config": {
            "width_coefficient": 1.8,
            "depth_coefficient": 2.6,
            "dropout_rate": 0.5,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "stackwise_kernel_sizes": [3, 3, 5, 3, 5, 5, 3],
            "stackwise_num_repeats": [1, 2, 2, 3, 3, 4, 1],
            "stackwise_input_filters": [32, 16, 24, 40, 80, 112, 192],
            "stackwise_output_filters": [16, 24, 40, 80, 112, 192, 320],
            "stackwise_expansion_ratios": [1, 6, 6, 6, 6, 6, 6],
            "stackwise_id_skip": True,
            "stackwise_strides": [1, 2, 2, 2, 1, 2, 1],
            "stackwise_squeeze_and_excite_ratios": [
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
            ],
        },
    },
    "efficientnetv1_b7": {
        "metadata": {
            "description": DESCRIPTION,
        },
        "class_name": "keras_cv.models>EfficientNetV1Backbone",
        "config": {
            "width_coefficient": 2.0,
            "depth_coefficient": 3.1,
            "dropout_rate": 0.5,
            "drop_connect_rate": 0.2,
            "depth_divisor": 8,
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "stackwise_kernel_sizes": [3, 3, 5, 3, 5, 5, 3],
            "stackwise_num_repeats": [1, 2, 2, 3, 3, 4, 1],
            "stackwise_input_filters": [32, 16, 24, 40, 80, 112, 192],
            "stackwise_output_filters": [16, 24, 40, 80, 112, 192, 320],
            "stackwise_expansion_ratios": [1, 6, 6, 6, 6, 6, 6],
            "stackwise_id_skip": True,
            "stackwise_strides": [1, 2, 2, 2, 1, 2, 1],
            "stackwise_squeeze_and_excite_ratios": [
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
            ],
        },
    },
}

backbone_presets_with_weights = {}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
