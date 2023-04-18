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

"""ResNetV1 model preset configurations."""

backbone_presets_no_weights = {
    "efficientnetv2-s": {
        "metadata": {
            "description": ("EfficientNetv2 model"),
        },
        "class_name": "keras_cv.models>ResNetBackbone",
        "config": [
            {
                "kernel_size": 3,
                "num_repeat": 2,
                "input_filters": 24,
                "output_filters": 24,
                "expand_ratio": 1,
                "se_ratio": 0.0,
                "strides": 1,
                "conv_type": 1,
            },
            {
                "kernel_size": 3,
                "num_repeat": 4,
                "input_filters": 24,
                "output_filters": 48,
                "expand_ratio": 4,
                "se_ratio": 0.0,
                "strides": 2,
                "conv_type": 1,
            },
            {
                "conv_type": 1,
                "expand_ratio": 4,
                "input_filters": 48,
                "kernel_size": 3,
                "num_repeat": 4,
                "output_filters": 64,
                "se_ratio": 0,
                "strides": 2,
            },
            {
                "conv_type": 0,
                "expand_ratio": 4,
                "input_filters": 64,
                "kernel_size": 3,
                "num_repeat": 6,
                "output_filters": 128,
                "se_ratio": 0.25,
                "strides": 2,
            },
            {
                "conv_type": 0,
                "expand_ratio": 6,
                "input_filters": 128,
                "kernel_size": 3,
                "num_repeat": 9,
                "output_filters": 160,
                "se_ratio": 0.25,
                "strides": 1,
            },
            {
                "conv_type": 0,
                "expand_ratio": 6,
                "input_filters": 160,
                "kernel_size": 3,
                "num_repeat": 15,
                "output_filters": 256,
                "se_ratio": 0.25,
                "strides": 2,
            },
        ],
    },
    "efficientnetv2-m": {
        "metadata": {
            "description": (
                "ResNet model with 34 layers where the batch normalization "
                "and ReLU activation are applied after the convolution layers "
                "(v1 style)."
            ),
        },
        "class_name": "keras_cv.models>ResNetBackbone",
        "config": [
            {
                "kernel_size": 3,
                "num_repeat": 3,
                "input_filters": 24,
                "output_filters": 24,
                "expand_ratio": 1,
                "se_ratio": 0,
                "strides": 1,
                "conv_type": 1,
            },
            {
                "kernel_size": 3,
                "num_repeat": 5,
                "input_filters": 24,
                "output_filters": 48,
                "expand_ratio": 4,
                "se_ratio": 0,
                "strides": 2,
                "conv_type": 1,
            },
            {
                "kernel_size": 3,
                "num_repeat": 5,
                "input_filters": 48,
                "output_filters": 80,
                "expand_ratio": 4,
                "se_ratio": 0,
                "strides": 2,
                "conv_type": 1,
            },
            {
                "kernel_size": 3,
                "num_repeat": 7,
                "input_filters": 80,
                "output_filters": 160,
                "expand_ratio": 4,
                "se_ratio": 0.25,
                "strides": 2,
                "conv_type": 0,
            },
            {
                "kernel_size": 3,
                "num_repeat": 14,
                "input_filters": 160,
                "output_filters": 176,
                "expand_ratio": 6,
                "se_ratio": 0.25,
                "strides": 1,
                "conv_type": 0,
            },
            {
                "kernel_size": 3,
                "num_repeat": 18,
                "input_filters": 176,
                "output_filters": 304,
                "expand_ratio": 6,
                "se_ratio": 0.25,
                "strides": 2,
                "conv_type": 0,
            },
            {
                "kernel_size": 3,
                "num_repeat": 5,
                "input_filters": 304,
                "output_filters": 512,
                "expand_ratio": 6,
                "se_ratio": 0.25,
                "strides": 1,
                "conv_type": 0,
            },
        ],
    },
    "efficientnetv2-l": {
        "metadata": {
            "description": (
                "ResNet model with 50 layers where the batch normalization "
                "and ReLU activation are applied after the convolution layers "
                "(v1 style)."
            ),
        },
        "class_name": "keras_cv.models>ResNetBackbone",
        "config": [
            {
                "kernel_size": 3,
                "num_repeat": 4,
                "input_filters": 32,
                "output_filters": 32,
                "expand_ratio": 1,
                "se_ratio": 0,
                "strides": 1,
                "conv_type": 1,
            },
            {
                "kernel_size": 3,
                "num_repeat": 7,
                "input_filters": 32,
                "output_filters": 64,
                "expand_ratio": 4,
                "se_ratio": 0,
                "strides": 2,
                "conv_type": 1,
            },
            {
                "kernel_size": 3,
                "num_repeat": 7,
                "input_filters": 64,
                "output_filters": 96,
                "expand_ratio": 4,
                "se_ratio": 0,
                "strides": 2,
                "conv_type": 1,
            },
            {
                "kernel_size": 3,
                "num_repeat": 10,
                "input_filters": 96,
                "output_filters": 192,
                "expand_ratio": 4,
                "se_ratio": 0.25,
                "strides": 2,
                "conv_type": 0,
            },
            {
                "kernel_size": 3,
                "num_repeat": 19,
                "input_filters": 192,
                "output_filters": 224,
                "expand_ratio": 6,
                "se_ratio": 0.25,
                "strides": 1,
                "conv_type": 0,
            },
            {
                "kernel_size": 3,
                "num_repeat": 25,
                "input_filters": 224,
                "output_filters": 384,
                "expand_ratio": 6,
                "se_ratio": 0.25,
                "strides": 2,
                "conv_type": 0,
            },
            {
                "kernel_size": 3,
                "num_repeat": 7,
                "input_filters": 384,
                "output_filters": 640,
                "expand_ratio": 6,
                "se_ratio": 0.25,
                "strides": 1,
                "conv_type": 0,
            },
        ],
    },
    "efficientnetv2-b0": {
        "metadata": {
            "description": (
                "ResNet model with 101 layers where the batch normalization "
                "and ReLU activation are applied after the convolution layers "
                "(v1 style)."
            ),
        },
        "class_name": "keras_cv.models>ResNetBackbone",
        "config": [
            {
                "kernel_size": 3,
                "num_repeat": 1,
                "input_filters": 32,
                "output_filters": 16,
                "expand_ratio": 1,
                "se_ratio": 0,
                "strides": 1,
                "conv_type": 1,
            },
            {
                "kernel_size": 3,
                "num_repeat": 2,
                "input_filters": 16,
                "output_filters": 32,
                "expand_ratio": 4,
                "se_ratio": 0,
                "strides": 2,
                "conv_type": 1,
            },
            {
                "kernel_size": 3,
                "num_repeat": 2,
                "input_filters": 32,
                "output_filters": 48,
                "expand_ratio": 4,
                "se_ratio": 0,
                "strides": 2,
                "conv_type": 1,
            },
            {
                "kernel_size": 3,
                "num_repeat": 3,
                "input_filters": 48,
                "output_filters": 96,
                "expand_ratio": 4,
                "se_ratio": 0.25,
                "strides": 2,
                "conv_type": 0,
            },
            {
                "kernel_size": 3,
                "num_repeat": 5,
                "input_filters": 96,
                "output_filters": 112,
                "expand_ratio": 6,
                "se_ratio": 0.25,
                "strides": 1,
                "conv_type": 0,
            },
            {
                "kernel_size": 3,
                "num_repeat": 8,
                "input_filters": 112,
                "output_filters": 192,
                "expand_ratio": 6,
                "se_ratio": 0.25,
                "strides": 2,
                "conv_type": 0,
            },
        ],
    },
    "efficientnetv2-b1": {
        "metadata": {
            "description": (
                "ResNet model with 152 layers where the batch normalization "
                "and ReLU activation are applied after the convolution layers "
                "(v1 style)."
            ),
        },
        "class_name": "keras_cv.models>ResNetBackbone",
        "config": [
            {
                "kernel_size": 3,
                "num_repeat": 1,
                "input_filters": 32,
                "output_filters": 16,
                "expand_ratio": 1,
                "se_ratio": 0,
                "strides": 1,
                "conv_type": 1,
            },
            {
                "kernel_size": 3,
                "num_repeat": 2,
                "input_filters": 16,
                "output_filters": 32,
                "expand_ratio": 4,
                "se_ratio": 0,
                "strides": 2,
                "conv_type": 1,
            },
            {
                "kernel_size": 3,
                "num_repeat": 2,
                "input_filters": 32,
                "output_filters": 48,
                "expand_ratio": 4,
                "se_ratio": 0,
                "strides": 2,
                "conv_type": 1,
            },
            {
                "kernel_size": 3,
                "num_repeat": 3,
                "input_filters": 48,
                "output_filters": 96,
                "expand_ratio": 4,
                "se_ratio": 0.25,
                "strides": 2,
                "conv_type": 0,
            },
            {
                "kernel_size": 3,
                "num_repeat": 5,
                "input_filters": 96,
                "output_filters": 112,
                "expand_ratio": 6,
                "se_ratio": 0.25,
                "strides": 1,
                "conv_type": 0,
            },
            {
                "kernel_size": 3,
                "num_repeat": 8,
                "input_filters": 112,
                "output_filters": 192,
                "expand_ratio": 6,
                "se_ratio": 0.25,
                "strides": 2,
                "conv_type": 0,
            },
        ],
    },
    "efficientnetv2-b2": {
        "metadata": {
            "description": (
                "ResNet model with 152 layers where the batch normalization "
                "and ReLU activation are applied after the convolution layers "
                "(v1 style)."
            ),
        },
        "class_name": "keras_cv.models>ResNetBackbone",
        "config": [
            {
                "kernel_size": 3,
                "num_repeat": 1,
                "input_filters": 32,
                "output_filters": 16,
                "expand_ratio": 1,
                "se_ratio": 0,
                "strides": 1,
                "conv_type": 1,
            },
            {
                "kernel_size": 3,
                "num_repeat": 2,
                "input_filters": 16,
                "output_filters": 32,
                "expand_ratio": 4,
                "se_ratio": 0,
                "strides": 2,
                "conv_type": 1,
            },
            {
                "kernel_size": 3,
                "num_repeat": 2,
                "input_filters": 32,
                "output_filters": 48,
                "expand_ratio": 4,
                "se_ratio": 0,
                "strides": 2,
                "conv_type": 1,
            },
            {
                "kernel_size": 3,
                "num_repeat": 3,
                "input_filters": 48,
                "output_filters": 96,
                "expand_ratio": 4,
                "se_ratio": 0.25,
                "strides": 2,
                "conv_type": 0,
            },
            {
                "kernel_size": 3,
                "num_repeat": 5,
                "input_filters": 96,
                "output_filters": 112,
                "expand_ratio": 6,
                "se_ratio": 0.25,
                "strides": 1,
                "conv_type": 0,
            },
            {
                "kernel_size": 3,
                "num_repeat": 8,
                "input_filters": 112,
                "output_filters": 192,
                "expand_ratio": 6,
                "se_ratio": 0.25,
                "strides": 2,
                "conv_type": 0,
            },
        ],
    },
    "efficientnetv2-b3": {
        "metadata": {
            "description": (
                "ResNet model with 152 layers where the batch normalization "
                "and ReLU activation are applied after the convolution layers "
                "(v1 style)."
            ),
        },
        "class_name": "keras_cv.models>ResNetBackbone",
        "config": [
            {
                "kernel_size": 3,
                "num_repeat": 1,
                "input_filters": 32,
                "output_filters": 16,
                "expand_ratio": 1,
                "se_ratio": 0,
                "strides": 1,
                "conv_type": 1,
            },
            {
                "kernel_size": 3,
                "num_repeat": 2,
                "input_filters": 16,
                "output_filters": 32,
                "expand_ratio": 4,
                "se_ratio": 0,
                "strides": 2,
                "conv_type": 1,
            },
            {
                "kernel_size": 3,
                "num_repeat": 2,
                "input_filters": 32,
                "output_filters": 48,
                "expand_ratio": 4,
                "se_ratio": 0,
                "strides": 2,
                "conv_type": 1,
            },
            {
                "kernel_size": 3,
                "num_repeat": 3,
                "input_filters": 48,
                "output_filters": 96,
                "expand_ratio": 4,
                "se_ratio": 0.25,
                "strides": 2,
                "conv_type": 0,
            },
            {
                "kernel_size": 3,
                "num_repeat": 5,
                "input_filters": 96,
                "output_filters": 112,
                "expand_ratio": 6,
                "se_ratio": 0.25,
                "strides": 1,
                "conv_type": 0,
            },
            {
                "kernel_size": 3,
                "num_repeat": 8,
                "input_filters": 112,
                "output_filters": 192,
                "expand_ratio": 6,
                "se_ratio": 0.25,
                "strides": 2,
                "conv_type": 0,
            },
        ],
    },
}

backbone_presets_with_weights = {}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
