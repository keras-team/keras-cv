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

SMALL_BLOCK_ARGS = [
    {
        "kernel_size": 3,
        "num_repeat": 2,
        "input_filters": 24,
        "output_filters": 24,
        "expand_ratio": 1,
        "se_ratio": 0.0,
        "strides": 1,
        "conv_type": "fused_mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 4,
        "input_filters": 24,
        "output_filters": 48,
        "expand_ratio": 4,
        "se_ratio": 0.0,
        "strides": 2,
        "conv_type": "fused_mb_conv",
    },
    {
        "conv_type": "fused_mb_conv",
        "expand_ratio": 4,
        "input_filters": 48,
        "kernel_size": 3,
        "num_repeat": 4,
        "output_filters": 64,
        "se_ratio": 0,
        "strides": 2,
    },
    {
        "conv_type": "mb_conv",
        "expand_ratio": 4,
        "input_filters": 64,
        "kernel_size": 3,
        "num_repeat": 6,
        "output_filters": 128,
        "se_ratio": 0.25,
        "strides": 2,
    },
    {
        "conv_type": "mb_conv",
        "expand_ratio": 6,
        "input_filters": 128,
        "kernel_size": 3,
        "num_repeat": 9,
        "output_filters": 160,
        "se_ratio": 0.25,
        "strides": 1,
    },
    {
        "conv_type": "mb_conv",
        "expand_ratio": 6,
        "input_filters": 160,
        "kernel_size": 3,
        "num_repeat": 15,
        "output_filters": 256,
        "se_ratio": 0.25,
        "strides": 2,
    },
]

MEDIUM_BLOCK_ARGS = [
    {
        "kernel_size": 3,
        "num_repeat": 3,
        "input_filters": 24,
        "output_filters": 24,
        "expand_ratio": 1,
        "se_ratio": 0,
        "strides": 1,
        "conv_type": "fused_mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 5,
        "input_filters": 24,
        "output_filters": 48,
        "expand_ratio": 4,
        "se_ratio": 0,
        "strides": 2,
        "conv_type": "fused_mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 5,
        "input_filters": 48,
        "output_filters": 80,
        "expand_ratio": 4,
        "se_ratio": 0,
        "strides": 2,
        "conv_type": "fused_mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 7,
        "input_filters": 80,
        "output_filters": 160,
        "expand_ratio": 4,
        "se_ratio": 0.25,
        "strides": 2,
        "conv_type": "mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 14,
        "input_filters": 160,
        "output_filters": 176,
        "expand_ratio": 6,
        "se_ratio": 0.25,
        "strides": 1,
        "conv_type": "mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 18,
        "input_filters": 176,
        "output_filters": 304,
        "expand_ratio": 6,
        "se_ratio": 0.25,
        "strides": 2,
        "conv_type": "mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 5,
        "input_filters": 304,
        "output_filters": 512,
        "expand_ratio": 6,
        "se_ratio": 0.25,
        "strides": 1,
        "conv_type": "mb_conv",
    },
]

LARGE_BLOCK_ARGS = [
    {
        "kernel_size": 3,
        "num_repeat": 4,
        "input_filters": 32,
        "output_filters": 32,
        "expand_ratio": 1,
        "se_ratio": 0,
        "strides": 1,
        "conv_type": "fused_mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 7,
        "input_filters": 32,
        "output_filters": 64,
        "expand_ratio": 4,
        "se_ratio": 0,
        "strides": 2,
        "conv_type": "fused_mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 7,
        "input_filters": 64,
        "output_filters": 96,
        "expand_ratio": 4,
        "se_ratio": 0,
        "strides": 2,
        "conv_type": "fused_mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 10,
        "input_filters": 96,
        "output_filters": 192,
        "expand_ratio": 4,
        "se_ratio": 0.25,
        "strides": 2,
        "conv_type": "mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 19,
        "input_filters": 192,
        "output_filters": 224,
        "expand_ratio": 6,
        "se_ratio": 0.25,
        "strides": 1,
        "conv_type": "mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 25,
        "input_filters": 224,
        "output_filters": 384,
        "expand_ratio": 6,
        "se_ratio": 0.25,
        "strides": 2,
        "conv_type": "mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 7,
        "input_filters": 384,
        "output_filters": 640,
        "expand_ratio": 6,
        "se_ratio": 0.25,
        "strides": 1,
        "conv_type": "mb_conv",
    },
]

B0_BLOCK_ARGS = [
    {
        "kernel_size": 3,
        "num_repeat": 1,
        "input_filters": 32,
        "output_filters": 16,
        "expand_ratio": 1,
        "se_ratio": 0,
        "strides": 1,
        "conv_type": "fused_mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 2,
        "input_filters": 16,
        "output_filters": 32,
        "expand_ratio": 4,
        "se_ratio": 0,
        "strides": 2,
        "conv_type": "fused_mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 2,
        "input_filters": 32,
        "output_filters": 48,
        "expand_ratio": 4,
        "se_ratio": 0,
        "strides": 2,
        "conv_type": "fused_mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 3,
        "input_filters": 48,
        "output_filters": 96,
        "expand_ratio": 4,
        "se_ratio": 0.25,
        "strides": 2,
        "conv_type": "mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 5,
        "input_filters": 96,
        "output_filters": 112,
        "expand_ratio": 6,
        "se_ratio": 0.25,
        "strides": 1,
        "conv_type": "mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 8,
        "input_filters": 112,
        "output_filters": 192,
        "expand_ratio": 6,
        "se_ratio": 0.25,
        "strides": 2,
        "conv_type": "mb_conv",
    },
]

B1_BLOCK_ARGS = [
    {
        "kernel_size": 3,
        "num_repeat": 1,
        "input_filters": 32,
        "output_filters": 16,
        "expand_ratio": 1,
        "se_ratio": 0,
        "strides": 1,
        "conv_type": "fused_mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 2,
        "input_filters": 16,
        "output_filters": 32,
        "expand_ratio": 4,
        "se_ratio": 0,
        "strides": 2,
        "conv_type": "fused_mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 2,
        "input_filters": 32,
        "output_filters": 48,
        "expand_ratio": 4,
        "se_ratio": 0,
        "strides": 2,
        "conv_type": "fused_mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 3,
        "input_filters": 48,
        "output_filters": 96,
        "expand_ratio": 4,
        "se_ratio": 0.25,
        "strides": 2,
        "conv_type": "mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 5,
        "input_filters": 96,
        "output_filters": 112,
        "expand_ratio": 6,
        "se_ratio": 0.25,
        "strides": 1,
        "conv_type": "mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 8,
        "input_filters": 112,
        "output_filters": 192,
        "expand_ratio": 6,
        "se_ratio": 0.25,
        "strides": 2,
        "conv_type": "mb_conv",
    },
]

B2_BLOCK_ARGS = [
    {
        "kernel_size": 3,
        "num_repeat": 1,
        "input_filters": 32,
        "output_filters": 16,
        "expand_ratio": 1,
        "se_ratio": 0,
        "strides": 1,
        "conv_type": "fused_mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 2,
        "input_filters": 16,
        "output_filters": 32,
        "expand_ratio": 4,
        "se_ratio": 0,
        "strides": 2,
        "conv_type": "fused_mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 2,
        "input_filters": 32,
        "output_filters": 48,
        "expand_ratio": 4,
        "se_ratio": 0,
        "strides": 2,
        "conv_type": "fused_mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 3,
        "input_filters": 48,
        "output_filters": 96,
        "expand_ratio": 4,
        "se_ratio": 0.25,
        "strides": 2,
        "conv_type": "mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 5,
        "input_filters": 96,
        "output_filters": 112,
        "expand_ratio": 6,
        "se_ratio": 0.25,
        "strides": 1,
        "conv_type": "mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 8,
        "input_filters": 112,
        "output_filters": 192,
        "expand_ratio": 6,
        "se_ratio": 0.25,
        "strides": 2,
        "conv_type": "mb_conv",
    },
]

B3_BLOCK_ARGS = [
    {
        "kernel_size": 3,
        "num_repeat": 1,
        "input_filters": 32,
        "output_filters": 16,
        "expand_ratio": 1,
        "se_ratio": 0,
        "strides": 1,
        "conv_type": "fused_mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 2,
        "input_filters": 16,
        "output_filters": 32,
        "expand_ratio": 4,
        "se_ratio": 0,
        "strides": 2,
        "conv_type": "fused_mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 2,
        "input_filters": 32,
        "output_filters": 48,
        "expand_ratio": 4,
        "se_ratio": 0,
        "strides": 2,
        "conv_type": "fused_mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 3,
        "input_filters": 48,
        "output_filters": 96,
        "expand_ratio": 4,
        "se_ratio": 0.25,
        "strides": 2,
        "conv_type": "mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 5,
        "input_filters": 96,
        "output_filters": 112,
        "expand_ratio": 6,
        "se_ratio": 0.25,
        "strides": 1,
        "conv_type": "mb_conv",
    },
    {
        "kernel_size": 3,
        "num_repeat": 8,
        "input_filters": 112,
        "output_filters": 192,
        "expand_ratio": 6,
        "se_ratio": 0.25,
        "strides": 2,
        "conv_type": "mb_conv",
    },
]

backbone_presets_no_weights = {
    "efficientnetv2-s": {
        "model_name": "efficientnetv2-s",
        "metadata": {
            "description": "The EfficientNet small architecture.  In this "
            "variant of the EfficientNet architecture, there are "
            f"{len(SMALL_BLOCK_ARGS)} convolutional blocks."
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "model_name": "efficientnetv2-s",
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
            "blocks_args": SMALL_BLOCK_ARGS,
        },
    },
    "efficientnetv2-m": {
        "metadata": {
            "description": "The EfficientNet medium architecture.  In this "
            "variant of the EfficientNet architecture, there are "
            f"{len(MEDIUM_BLOCK_ARGS)} convolutional blocks."
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "model_name": "efficientnetv2-m",
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
            "blocks_args": MEDIUM_BLOCK_ARGS,
        },
    },
    "efficientnetv2-l": {
        "metadata": {
            "description": "The EfficientNet medium architecture.  In this "
            "variant of the EfficientNet architecture, there are "
            f"{len(LARGE_BLOCK_ARGS)} convolutional blocks."
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "model_name": "efficientnetv2-l",
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
            "blocks_args": LARGE_BLOCK_ARGS,
        },
    },
    "efficientnetv2-b0": {
        "metadata": {
            "description": "The EfficientNet B0 architecture.  In this "
            "variant of the EfficientNet architecture, there are "
            f"{len(B0_BLOCK_ARGS)} convolutional blocks. As with all of the B "
            "style EfficientNet variants, the number of filters in each "
            "convolutional block is scaled by a `width_coefficient` and a "
            "`depth_coefficient`.  Please see the GitHub source to find the "
            "specific values for both the `width_coefficient` and "
            "`depth_coefficient`"
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "model_name": "efficientnetv2-b0",
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
            "blocks_args": B0_BLOCK_ARGS,
        },
    },
    "efficientnetv2-b1": {
        "metadata": {
            "description": "The EfficientNet B1 architecture.  In this "
            "variant of the EfficientNet architecture, there are "
            f"{len(B1_BLOCK_ARGS)} convolutional blocks. As with all of the B "
            "style EfficientNet variants, the number of filters in each "
            "convolutional block is scaled by a `width_coefficient` and a "
            "`depth_coefficient`.  Please see the GitHub source to find the "
            "specific values for both the `width_coefficient` and "
            "`depth_coefficient`"
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "model_name": "efficientnetv2-b1",
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
            "blocks_args": B1_BLOCK_ARGS,
        },
    },
    "efficientnetv2-b2": {
        "metadata": {
            "description": "The EfficientNet B2 architecture.  In this "
            "variant of the EfficientNet architecture, there are "
            f"{len(B2_BLOCK_ARGS)} convolutional blocks. As with all of the B "
            "style EfficientNet variants, the number of filters in each "
            "convolutional block is scaled by a `width_coefficient` and a "
            "`depth_coefficient`.  Please see the GitHub source to find the "
            "specific values for both the `width_coefficient` and "
            "`depth_coefficient`"
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "model_name": "efficientnetv2-b2",
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
            "blocks_args": B2_BLOCK_ARGS,
        },
    },
    "efficientnetv2-b3": {
        "metadata": {
            "description": "The EfficientNet B3 architecture.  In this "
            "variant of the EfficientNet architecture, there are "
            f"{len(B3_BLOCK_ARGS)} convolutional blocks. As with all of the B "
            "style EfficientNet variants, the number of filters in each "
            "convolutional block is scaled by a `width_coefficient` and a "
            "`depth_coefficient`.  Please see the GitHub source to find the "
            "specific values for both the `width_coefficient` and "
            "`depth_coefficient`"
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "model_name": "efficientnetv2-b3",
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
            "blocks_args": B3_BLOCK_ARGS,
        },
    },
}

backbone_presets_with_weights = {
    "efficientnetv2-s_imagenet": {
        "metadata": {
            "description": DESCRIPTION,
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "model_name": "efficientnetv2-s",
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
            "blocks_args": SMALL_BLOCK_ARGS,
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/efficientnetv2s/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "80555436ea49100893552614b4dce98de461fa3b6c14f8132673817d28c83654",  # noqa: E501
    },
    "efficientnetv2-b0_imagenet": {
        "metadata": {
            "description": "EfficientNetv2 model",
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "model_name": "efficientnetv2-b0",
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
            "blocks_args": B0_BLOCK_ARGS,
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/efficientnetv2b0/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "ac95f13a8ad1cee41184fc16fd0eb769f7c5b3131151c6abf7fcee5cc3d09bc8",  # noqa: E501
    },
    "efficientnetv2-b1_imagenet": {
        "metadata": {
            "description": DESCRIPTION,
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "model_name": "efficientnetv2-b1",
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
            "blocks_args": B1_BLOCK_ARGS,
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/efficientnetv2b1/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "82da111f8411f47e3f5eef090da76340f38e222f90a08bead53662f2ebafb01c",  # noqa: E501
    },
    "efficientnetv2-b2_imagenet": {
        "metadata": {
            "description": DESCRIPTION,
        },
        "class_name": "keras_cv.models>EfficientNetV2Backbone",
        "config": {
            "model_name": "efficientnetv2-b2",
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
            "blocks_args": B2_BLOCK_ARGS,
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/efficientnetv2b2/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "02d12c9d1589b540b4e84ffdb54ff30c96099bd59e311a85ddc7180efc65e955",  # noqa: E501
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
