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
            "description": "The EfficientNet small architecture.  In this "
            "variant of the EfficientNet architecture, there are "
            f"6 convolutional blocks."
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
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "kernel_sizes": [3, 3, 3, 3, 3, 3],
            "num_repeats": [2, 4, 4, 6, 9, 15],
            "input_filters": [24, 24, 48, 64, 128, 160],
            "output_filters": [24, 48, 64, 128, 160, 256],
            "expand_ratios": [1, 4, 4, 4, 6, 6],
            "se_ratios": [0.0, 0.0, 0, 0.25, 0.25, 0.25],
            "strides": [1, 2, 2, 2, 1, 2],
            "conv_types": [
                "fused_mb_conv",
                "fused_mb_conv",
                "fused_mb_conv",
                "mb_conv",
                "mb_conv",
                "mb_conv",
            ],
        },
    },
    "efficientnetv2-m": {
        "metadata": {
            "description": "The EfficientNet medium architecture.  In this "
            "variant of the EfficientNet architecture, there are "
            "7 convolutional blocks."
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
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "kernel_sizes": [3, 3, 3, 3, 3, 3, 3],
            "num_repeats": [3, 5, 5, 7, 14, 18, 5],
            "input_filters": [24, 24, 48, 80, 160, 176, 304],
            "output_filters": [24, 48, 80, 160, 176, 304, 512],
            "expand_ratios": [1, 4, 4, 4, 6, 6, 6],
            "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25, 0.25],
            "strides": [1, 2, 2, 2, 1, 2, 1],
            "conv_types": [
                "fused_mb_conv",
                "fused_mb_conv",
                "fused_mb_conv",
                "mb_conv",
                "mb_conv",
                "mb_conv",
                "mb_conv",
            ],
        },
    },
    "efficientnetv2-l": {
        "metadata": {
            "description": "The EfficientNet medium architecture.  In this "
            "variant of the EfficientNet architecture, there are "
            "7 convolutional blocks.  The primary distinction between the "
            "EfficientNet large architecture and the EfficientNet medium lies "
            "in the number of filters contained in each layer, and the number "
            "of times each block is repeated."
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
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "kernel_sizes": [3, 3, 3, 3, 3, 3, 3],
            "num_repeats": [4, 7, 7, 10, 19, 25, 7],
            "input_filters": [32, 32, 64, 96, 192, 224, 384],
            "output_filters": [32, 64, 96, 192, 224, 384, 640],
            "expand_ratios": [1, 4, 4, 4, 6, 6, 6],
            "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25, 0.25],
            "strides": [1, 2, 2, 2, 1, 2, 1],
            "conv_types": [
                "fused_mb_conv",
                "fused_mb_conv",
                "fused_mb_conv",
                "mb_conv",
                "mb_conv",
                "mb_conv",
                "mb_conv",
            ],
        },
    },
    "efficientnetv2-b0": {
        "metadata": {
            "description": "The EfficientNet B0 architecture.  In this "
            "variant of the EfficientNet architecture, there are "
            "6 convolutional blocks. As with all of the B "
            "style EfficientNet variants, the number of filters in each "
            "convolutional block is scaled by `width_coefficient=1.0` and "
            "`depth_coefficient=1.0`."
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
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "kernel_sizes": [3, 3, 3, 3, 3, 3],
            "num_repeats": [1, 2, 2, 3, 5, 8],
            "input_filters": [32, 16, 32, 48, 96, 112],
            "output_filters": [16, 32, 48, 96, 112, 192],
            "expand_ratios": [1, 4, 4, 4, 6, 6],
            "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25],
            "strides": [1, 2, 2, 2, 1, 2],
            "conv_types": [
                "fused_mb_conv",
                "fused_mb_conv",
                "fused_mb_conv",
                "mb_conv",
                "mb_conv",
                "mb_conv",
            ],
        },
    },
    "efficientnetv2-b1": {
        "metadata": {
            "description": "The EfficientNet B1 architecture.  In this "
            "variant of the EfficientNet architecture, there are "
            f"6 convolutional blocks. As with all of the B "
            "style EfficientNet variants, the number of filters in each "
            "convolutional block is scaled by `width_coefficient=1.0` and a "
            "`depth_coefficient=1.1`."
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
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "kernel_sizes": [3, 3, 3, 3, 3, 3],
            "num_repeats": [1, 2, 2, 3, 5, 8],
            "input_filters": [32, 16, 32, 48, 96, 112],
            "output_filters": [16, 32, 48, 96, 112, 192],
            "expand_ratios": [1, 4, 4, 4, 6, 6],
            "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25],
            "strides": [1, 2, 2, 2, 1, 2],
            "conv_types": [
                "fused_mb_conv",
                "fused_mb_conv",
                "fused_mb_conv",
                "mb_conv",
                "mb_conv",
                "mb_conv",
            ],
        },
    },
    "efficientnetv2-b2": {
        "metadata": {
            "description": "The EfficientNet B2 architecture.  In this "
            "variant of the EfficientNet architecture, there are "
            "6 convolutional blocks. As with all of the B "
            "style EfficientNet variants, the number of filters in each "
            "convolutional block is scaled by `width_coefficient=1.1` and "
            "`depth_coefficient=1.2`."
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
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "kernel_sizes": [3, 3, 3, 3, 3, 3],
            "num_repeats": [1, 2, 2, 3, 5, 8],
            "input_filters": [32, 16, 32, 48, 96, 112],
            "output_filters": [16, 32, 48, 96, 112, 192],
            "expand_ratios": [1, 4, 4, 4, 6, 6],
            "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25],
            "strides": [1, 2, 2, 2, 1, 2],
            "conv_types": [
                "fused_mb_conv",
                "fused_mb_conv",
                "fused_mb_conv",
                "mb_conv",
                "mb_conv",
                "mb_conv",
            ],
        },
    },
    "efficientnetv2-b3": {
        "metadata": {
            "description": "The EfficientNet B3 architecture.  In this "
            "variant of the EfficientNet architecture, there are "
            f"6 convolutional blocks. As with all of the B "
            "style EfficientNet variants, the number of filters in each "
            "convolutional block is scaled by `width_coefficient=1.2` and "
            "`depth_coefficient=1.4`."
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
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "kernel_sizes": [3, 3, 3, 3, 3, 3],
            "num_repeats": [1, 2, 2, 3, 5, 8],
            "input_filters": [32, 16, 32, 48, 96, 112],
            "output_filters": [16, 32, 48, 96, 112, 192],
            "expand_ratios": [1, 4, 4, 4, 6, 6],
            "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25],
            "strides": [1, 2, 2, 2, 1, 2],
            "conv_types": [
                "fused_mb_conv",
                "fused_mb_conv",
                "fused_mb_conv",
                "mb_conv",
                "mb_conv",
                "mb_conv",
            ],
        },
    },
}

backbone_presets_with_weights = {
    "efficientnetv2-s_imagenet": {
        "metadata": {
            "description": {
                "description": "The EfficientNet small architecture.  In this "
                "variant of the EfficientNet architecture, there are "
                f"6 convolutional blocks. Weights are "
                "initialized to pretrained imagenet classification weights."
                "Published weights are capable of scoring 83.9%	top 1 accuracy and "
                "96.7% top 5 accuracy on imagenet."
            },
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
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "kernel_sizes": [3, 3, 3, 3, 3, 3],
            "num_repeats": [2, 4, 4, 6, 9, 15],
            "input_filters": [24, 24, 48, 64, 128, 160],
            "output_filters": [24, 48, 64, 128, 160, 256],
            "expand_ratios": [1, 4, 4, 4, 6, 6],
            "se_ratios": [0.0, 0.0, 0, 0.25, 0.25, 0.25],
            "strides": [1, 2, 2, 2, 1, 2],
            "conv_types": [
                "fused_mb_conv",
                "fused_mb_conv",
                "fused_mb_conv",
                "mb_conv",
                "mb_conv",
                "mb_conv",
            ],
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/efficientnetv2s/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "80555436ea49100893552614b4dce98de461fa3b6c14f8132673817d28c83654",  # noqa: E501
    },
    "efficientnetv2-b0_imagenet": {
        "metadata": {
            "description": "The EfficientNet B0 architecture.  In this "
            "variant of the EfficientNet architecture, there are "
            f"6 convolutional blocks. As with all of the B "
            "style EfficientNet variants, the number of filters in each "
            "convolutional block is scaled by `width_coefficient=1.0` and "
            "`depth_coefficient=1.0`. Weights are "
            "initialized to pretrained imagenet classification weights. "
            "Published weights are capable of scoring 77.1%	top 1 accuracy and "
            "93.3% top 5 accuracy on imagenet."
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
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "kernel_sizes": [3, 3, 3, 3, 3, 3],
            "num_repeats": [1, 2, 2, 3, 5, 8],
            "input_filters": [32, 16, 32, 48, 96, 112],
            "output_filters": [16, 32, 48, 96, 112, 192],
            "expand_ratios": [1, 4, 4, 4, 6, 6],
            "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25],
            "strides": [1, 2, 2, 2, 1, 2],
            "conv_types": [
                "fused_mb_conv",
                "fused_mb_conv",
                "fused_mb_conv",
                "mb_conv",
                "mb_conv",
                "mb_conv",
            ],
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/efficientnetv2b0/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "ac95f13a8ad1cee41184fc16fd0eb769f7c5b3131151c6abf7fcee5cc3d09bc8",  # noqa: E501
    },
    "efficientnetv2-b1_imagenet": {
        "metadata": {
            "description": "The EfficientNet B1 architecture.  In this "
            "variant of the EfficientNet architecture, there are "
            f"6 convolutional blocks. As with all of the B "
            "style EfficientNet variants, the number of filters in each "
            "convolutional block is scaled by `width_coefficient=1.0` and "
            "`depth_coefficient=1.1`. Weights are "
            "initialized to pretrained imagenet classification weights."
            "Published weights are capable of scoring 79.1%	top 1 accuracy and "
            "94.4% top 5 accuracy on imagenet."
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
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "kernel_sizes": [3, 3, 3, 3, 3, 3],
            "num_repeats": [1, 2, 2, 3, 5, 8],
            "input_filters": [32, 16, 32, 48, 96, 112],
            "output_filters": [16, 32, 48, 96, 112, 192],
            "expand_ratios": [1, 4, 4, 4, 6, 6],
            "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25],
            "strides": [1, 2, 2, 2, 1, 2],
            "conv_types": [
                "fused_mb_conv",
                "fused_mb_conv",
                "fused_mb_conv",
                "mb_conv",
                "mb_conv",
                "mb_conv",
            ],
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/efficientnetv2b1/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "82da111f8411f47e3f5eef090da76340f38e222f90a08bead53662f2ebafb01c",  # noqa: E501
    },
    "efficientnetv2-b2_imagenet": {
        "metadata": {
            "description": "The EfficientNet B2 architecture.  In this "
            "variant of the EfficientNet architecture, there are "
            f"6 convolutional blocks. As with all of the B "
            "style EfficientNet variants, the number of filters in each "
            "convolutional block is scaled by `width_coefficient=1.1` and "
            "`depth_coefficient1.2`. Weights are initialized to pretrained "
            "imagenet classification weights."
            "Published weights are capable of scoring 80.1%	top 1 accuracy and "
            "94.9% top 5 accuracy on imagenet."
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
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "kernel_sizes": [3, 3, 3, 3, 3, 3],
            "num_repeats": [1, 2, 2, 3, 5, 8],
            "input_filters": [32, 16, 32, 48, 96, 112],
            "output_filters": [16, 32, 48, 96, 112, 192],
            "expand_ratios": [1, 4, 4, 4, 6, 6],
            "se_ratios": [0, 0, 0, 0.25, 0.25, 0.25],
            "strides": [1, 2, 2, 2, 1, 2],
            "conv_types": [
                "fused_mb_conv",
                "fused_mb_conv",
                "fused_mb_conv",
                "mb_conv",
                "mb_conv",
                "mb_conv",
            ],
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/efficientnetv2b2/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "02d12c9d1589b540b4e84ffdb54ff30c96099bd59e311a85ddc7180efc65e955",  # noqa: E501
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
