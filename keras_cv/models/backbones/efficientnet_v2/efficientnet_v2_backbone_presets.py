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
    "efficientnetv2_s": {
        "metadata": {
            "description": (
                "EfficientNet architecture with 6 convolutional blocks."
            ),
            "params": 20331360,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "class_name": "keras_cv>EfficientNetV2Backbone",
        "config": {
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "skip_connection_dropout": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "stackwise_kernel_sizes": [3, 3, 3, 3, 3, 3],
            "stackwise_num_repeats": [2, 4, 4, 6, 9, 15],
            "stackwise_input_filters": [24, 24, 48, 64, 128, 160],
            "stackwise_output_filters": [24, 48, 64, 128, 160, 256],
            "stackwise_expansion_ratios": [1, 4, 4, 4, 6, 6],
            "stackwise_squeeze_and_excite_ratios": [
                0.0,
                0.0,
                0,
                0.25,
                0.25,
                0.25,
            ],
            "stackwise_strides": [1, 2, 2, 2, 1, 2],
            "stackwise_conv_types": [
                "fused",
                "fused",
                "fused",
                "unfused",
                "unfused",
                "unfused",
            ],
        },
    },
    "efficientnetv2_m": {
        "metadata": {
            "description": (
                "EfficientNet architecture with 7 convolutional blocks."
            ),
            "params": 53150388,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "class_name": "keras_cv>EfficientNetV2Backbone",
        "config": {
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "skip_connection_dropout": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "stackwise_kernel_sizes": [3, 3, 3, 3, 3, 3, 3],
            "stackwise_num_repeats": [3, 5, 5, 7, 14, 18, 5],
            "stackwise_input_filters": [24, 24, 48, 80, 160, 176, 304],
            "stackwise_output_filters": [24, 48, 80, 160, 176, 304, 512],
            "stackwise_expansion_ratios": [1, 4, 4, 4, 6, 6, 6],
            "stackwise_squeeze_and_excite_ratios": [
                0,
                0,
                0,
                0.25,
                0.25,
                0.25,
                0.25,
            ],
            "stackwise_strides": [1, 2, 2, 2, 1, 2, 1],
            "stackwise_conv_types": [
                "fused",
                "fused",
                "fused",
                "unfused",
                "unfused",
                "unfused",
                "unfused",
            ],
        },
    },
    "efficientnetv2_l": {
        "metadata": {
            "description": (
                "EfficientNet architecture with 7 convolutional "
                "blocks, but more filters the in `efficientnetv2_m`."
            ),
            "params": 117746848,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "class_name": "keras_cv>EfficientNetV2Backbone",
        "config": {
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "skip_connection_dropout": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "stackwise_kernel_sizes": [3, 3, 3, 3, 3, 3, 3],
            "stackwise_num_repeats": [4, 7, 7, 10, 19, 25, 7],
            "stackwise_input_filters": [32, 32, 64, 96, 192, 224, 384],
            "stackwise_output_filters": [32, 64, 96, 192, 224, 384, 640],
            "stackwise_expansion_ratios": [1, 4, 4, 4, 6, 6, 6],
            "stackwise_squeeze_and_excite_ratios": [
                0,
                0,
                0,
                0.25,
                0.25,
                0.25,
                0.25,
            ],
            "stackwise_strides": [1, 2, 2, 2, 1, 2, 1],
            "stackwise_conv_types": [
                "fused",
                "fused",
                "fused",
                "unfused",
                "unfused",
                "unfused",
                "unfused",
            ],
        },
    },
    "efficientnetv2_b0": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 6 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.0` and `depth_coefficient=1.0`."
            ),
            "params": 5919312,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "class_name": "keras_cv>EfficientNetV2Backbone",
        "config": {
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "skip_connection_dropout": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "stackwise_kernel_sizes": [3, 3, 3, 3, 3, 3],
            "stackwise_num_repeats": [1, 2, 2, 3, 5, 8],
            "stackwise_input_filters": [32, 16, 32, 48, 96, 112],
            "stackwise_output_filters": [16, 32, 48, 96, 112, 192],
            "stackwise_expansion_ratios": [1, 4, 4, 4, 6, 6],
            "stackwise_squeeze_and_excite_ratios": [0, 0, 0, 0.25, 0.25, 0.25],
            "stackwise_strides": [1, 2, 2, 2, 1, 2],
            "stackwise_conv_types": [
                "fused",
                "fused",
                "fused",
                "unfused",
                "unfused",
                "unfused",
            ],
        },
    },
    "efficientnetv2_b1": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 6 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.0` and `depth_coefficient=1.1`."
            ),
            "params": 6931124,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "class_name": "keras_cv>EfficientNetV2Backbone",
        "config": {
            "width_coefficient": 1.0,
            "depth_coefficient": 1.1,
            "skip_connection_dropout": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "stackwise_kernel_sizes": [3, 3, 3, 3, 3, 3],
            "stackwise_num_repeats": [1, 2, 2, 3, 5, 8],
            "stackwise_input_filters": [32, 16, 32, 48, 96, 112],
            "stackwise_output_filters": [16, 32, 48, 96, 112, 192],
            "stackwise_expansion_ratios": [1, 4, 4, 4, 6, 6],
            "stackwise_squeeze_and_excite_ratios": [0, 0, 0, 0.25, 0.25, 0.25],
            "stackwise_strides": [1, 2, 2, 2, 1, 2],
            "stackwise_conv_types": [
                "fused",
                "fused",
                "fused",
                "unfused",
                "unfused",
                "unfused",
            ],
        },
    },
    "efficientnetv2_b2": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 6 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.1` and `depth_coefficient=1.2`."
            ),
            "params": 8769374,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "class_name": "keras_cv>EfficientNetV2Backbone",
        "config": {
            "width_coefficient": 1.1,
            "depth_coefficient": 1.2,
            "skip_connection_dropout": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "stackwise_kernel_sizes": [3, 3, 3, 3, 3, 3],
            "stackwise_num_repeats": [1, 2, 2, 3, 5, 8],
            "stackwise_input_filters": [32, 16, 32, 48, 96, 112],
            "stackwise_output_filters": [16, 32, 48, 96, 112, 192],
            "stackwise_expansion_ratios": [1, 4, 4, 4, 6, 6],
            "stackwise_squeeze_and_excite_ratios": [0, 0, 0, 0.25, 0.25, 0.25],
            "stackwise_strides": [1, 2, 2, 2, 1, 2],
            "stackwise_conv_types": [
                "fused",
                "fused",
                "fused",
                "unfused",
                "unfused",
                "unfused",
            ],
        },
    },
    "efficientnetv2_b3": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.2` and `depth_coefficient=1.4`."
            ),
            "params": 12930622,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "class_name": "keras_cv>EfficientNetV2Backbone",
        "config": {
            "width_coefficient": 1.2,
            "depth_coefficient": 1.4,
            "skip_connection_dropout": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "stackwise_kernel_sizes": [3, 3, 3, 3, 3, 3],
            "stackwise_num_repeats": [1, 2, 2, 3, 5, 8],
            "stackwise_input_filters": [32, 16, 32, 48, 96, 112],
            "stackwise_output_filters": [16, 32, 48, 96, 112, 192],
            "stackwise_expansion_ratios": [1, 4, 4, 4, 6, 6],
            "stackwise_squeeze_and_excite_ratios": [0, 0, 0, 0.25, 0.25, 0.25],
            "stackwise_strides": [1, 2, 2, 2, 1, 2],
            "stackwise_conv_types": [
                "fused",
                "fused",
                "fused",
                "unfused",
                "unfused",
                "unfused",
            ],
        },
    },
}

backbone_presets_with_weights = {
    "efficientnetv2_s_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet architecture with 6 convolutional "
                "blocks. Weights are initialized to pretrained imagenet "
                "classification weights.Published weights are capable of "
                "scoring 83.9%top 1 accuracy "
                "and 96.7% top 5 accuracy on imagenet."
            ),
            "params": 20331360,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "class_name": "keras_cv>EfficientNetV2Backbone",
        "config": {
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "skip_connection_dropout": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "stackwise_kernel_sizes": [3, 3, 3, 3, 3, 3],
            "stackwise_num_repeats": [2, 4, 4, 6, 9, 15],
            "stackwise_input_filters": [24, 24, 48, 64, 128, 160],
            "stackwise_output_filters": [24, 48, 64, 128, 160, 256],
            "stackwise_expansion_ratios": [1, 4, 4, 4, 6, 6],
            "stackwise_squeeze_and_excite_ratios": [
                0.0,
                0.0,
                0,
                0.25,
                0.25,
                0.25,
            ],
            "stackwise_strides": [1, 2, 2, 2, 1, 2],
            "stackwise_conv_types": [
                "fused",
                "fused",
                "fused",
                "unfused",
                "unfused",
                "unfused",
            ],
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/efficientnetv2s/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "80555436ea49100893552614b4dce98de461fa3b6c14f8132673817d28c83654",  # noqa: E501
    },
    "efficientnetv2_b0_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 6 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.0` and `depth_coefficient=1.0`. "
                "Weights are "
                "initialized to pretrained imagenet classification weights. "
                "Published weights are capable of scoring 77.1%	top 1 accuracy "
                "and 93.3% top 5 accuracy on imagenet."
            ),
            "params": 5919312,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "class_name": "keras_cv>EfficientNetV2Backbone",
        "config": {
            "width_coefficient": 1.0,
            "depth_coefficient": 1.0,
            "skip_connection_dropout": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "stackwise_kernel_sizes": [3, 3, 3, 3, 3, 3],
            "stackwise_num_repeats": [1, 2, 2, 3, 5, 8],
            "stackwise_input_filters": [32, 16, 32, 48, 96, 112],
            "stackwise_output_filters": [16, 32, 48, 96, 112, 192],
            "stackwise_expansion_ratios": [1, 4, 4, 4, 6, 6],
            "stackwise_squeeze_and_excite_ratios": [0, 0, 0, 0.25, 0.25, 0.25],
            "stackwise_strides": [1, 2, 2, 2, 1, 2],
            "stackwise_conv_types": [
                "fused",
                "fused",
                "fused",
                "unfused",
                "unfused",
                "unfused",
            ],
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/efficientnetv2b0/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "ac95f13a8ad1cee41184fc16fd0eb769f7c5b3131151c6abf7fcee5cc3d09bc8",  # noqa: E501
    },
    "efficientnetv2_b1_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 6 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.0` and `depth_coefficient=1.1`. "
                "Weights are "
                "initialized to pretrained imagenet classification weights."
                "Published weights are capable of scoring 79.1%	top 1 accuracy "
                "and 94.4% top 5 accuracy on imagenet."
            ),
            "params": 6931124,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "class_name": "keras_cv>EfficientNetV2Backbone",
        "config": {
            "width_coefficient": 1.0,
            "depth_coefficient": 1.1,
            "skip_connection_dropout": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "stackwise_kernel_sizes": [3, 3, 3, 3, 3, 3],
            "stackwise_num_repeats": [1, 2, 2, 3, 5, 8],
            "stackwise_input_filters": [32, 16, 32, 48, 96, 112],
            "stackwise_output_filters": [16, 32, 48, 96, 112, 192],
            "stackwise_expansion_ratios": [1, 4, 4, 4, 6, 6],
            "stackwise_squeeze_and_excite_ratios": [0, 0, 0, 0.25, 0.25, 0.25],
            "stackwise_strides": [1, 2, 2, 2, 1, 2],
            "stackwise_conv_types": [
                "fused",
                "fused",
                "fused",
                "unfused",
                "unfused",
                "unfused",
            ],
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/efficientnetv2b1/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "82da111f8411f47e3f5eef090da76340f38e222f90a08bead53662f2ebafb01c",  # noqa: E501
    },
    "efficientnetv2_b2_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 6 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.1` and `depth_coefficient=1.2`. "
                "Weights are initialized to pretrained "
                "imagenet classification weights."
                "Published weights are capable of scoring 80.1%	top 1 accuracy "
                "and 94.9% top 5 accuracy on imagenet."
            ),
            "params": 8769374,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "class_name": "keras_cv>EfficientNetV2Backbone",
        "config": {
            "width_coefficient": 1.1,
            "depth_coefficient": 1.2,
            "skip_connection_dropout": 0.2,
            "depth_divisor": 8,
            "min_depth": 8,
            "activation": "swish",
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "stackwise_kernel_sizes": [3, 3, 3, 3, 3, 3],
            "stackwise_num_repeats": [1, 2, 2, 3, 5, 8],
            "stackwise_input_filters": [32, 16, 32, 48, 96, 112],
            "stackwise_output_filters": [16, 32, 48, 96, 112, 192],
            "stackwise_expansion_ratios": [1, 4, 4, 4, 6, 6],
            "stackwise_squeeze_and_excite_ratios": [0, 0, 0, 0.25, 0.25, 0.25],
            "stackwise_strides": [1, 2, 2, 2, 1, 2],
            "stackwise_conv_types": [
                "fused",
                "fused",
                "fused",
                "unfused",
                "unfused",
                "unfused",
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
