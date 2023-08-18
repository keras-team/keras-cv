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
"""MobileNetV3 model preset configurations."""

backbone_presets_no_weights = {
    "mobilenet_v3_small": {
        "metadata": {
            "description": (
                "MobileNetV3 model with 14 layers where the batch "
                "normalization and hard-swish activation are applied after the "
                "convolution layers."
            ),
            "params": 933502,
            "official_name": "MobileNetV3",
            "path": "mobilenetv3",
        },
        "class_name": "keras_cv>MobileNetV3Backbone",
        "config": {
            "stackwise_expansion": [
                1,
                72.0 / 16,
                88.0 / 24,
                4,
                6,
                6,
                3,
                3,
                6,
                6,
                6,
            ],
            "stackwise_filters": [16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96],
            "stackwise_kernel_size": [3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5],
            "stackwise_stride": [2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1],
            "stackwise_se_ratio": [
                0.25,
                None,
                None,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
            ],
            "stackwise_activation": [
                "relu",
                "relu",
                "relu",
                "hard_swish",
                "hard_swish",
                "hard_swish",
                "hard_swish",
                "hard_swish",
                "hard_swish",
                "hard_swish",
                "hard_swish",
            ],
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "alpha": 1.0,
        },
    },
    "mobilenet_v3_large": {
        "metadata": {
            "description": (
                "MobileNetV3 model with 28 layers where the batch "
                "normalization and hard-swish activation are applied after the "
                "convolution layers."
            ),
            "params": 2994518,
            "official_name": "MobileNetV3",
            "path": "mobilenetv3",
        },
        "class_name": "keras_cv>MobileNetV3Backbone",
        "config": {
            "stackwise_expansion": [
                1,
                4,
                3,
                3,
                3,
                3,
                6,
                2.5,
                2.3,
                2.3,
                6,
                6,
                6,
                6,
                6,
            ],
            "stackwise_filters": [
                16,
                24,
                24,
                40,
                40,
                40,
                80,
                80,
                80,
                80,
                112,
                112,
                160,
                160,
                160,
            ],
            "stackwise_kernel_size": [
                3,
                3,
                3,
                5,
                5,
                5,
                3,
                3,
                3,
                3,
                3,
                3,
                5,
                5,
                5,
            ],
            "stackwise_stride": [1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1],
            "stackwise_se_ratio": [
                None,
                None,
                None,
                0.25,
                0.25,
                0.25,
                None,
                None,
                None,
                None,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
            ],
            "stackwise_activation": [
                "relu",
                "relu",
                "relu",
                "relu",
                "relu",
                "relu",
                "hard_swish",
                "hard_swish",
                "hard_swish",
                "hard_swish",
                "hard_swish",
                "hard_swish",
                "hard_swish",
                "hard_swish",
                "hard_swish",
            ],
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "alpha": 1.0,
        },
    },
}

backbone_presets_with_weights = {
    "mobilenet_v3_large_imagenet": {
        "metadata": {
            "description": (
                "MobileNetV3 model with 28 layers where the batch "
                "normalization and hard-swish activation are applied after the "
                "convolution layers. "
                "Pre-trained on the ImageNet 2012 classification task."
            ),
            "params": 2_994_518,
            "official_name": "MobileNetV3",
            "path": "mobilenetv3",
        },
        "class_name": "keras_cv>MobileNetV3Backbone",
        "config": backbone_presets_no_weights["mobilenet_v3_large"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/mobilenetv3/mobilenetv3_large_imagenet_backbone.h5",  # noqa: E501
        "weights_hash": "ec55ea2f4f4ee9a2ddf3ee8e2dd784e9d5732690c1fc5afc7e1b2a66703f3337",  # noqa: E501
    },
    "mobilenet_v3_small_imagenet": {
        "metadata": {
            "description": (
                "MobileNetV3 model with 28 layers where the batch "
                "normalization and hard-swish activation are applied after the "
                "convolution layers. "
                "Pre-trained on the ImageNet 2012 classification task."
            ),
            "params": 2_994_518,
            "official_name": "MobileNetV3",
            "path": "mobilenetv3",
        },
        "class_name": "keras_cv>MobileNetV3Backbone",
        "config": backbone_presets_no_weights["mobilenet_v3_small"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/mobilenetv3/mobilenetv3_small_imagenet_backbone.h5",  # noqa: E501
        "weights_hash": "592c2707edfc6c673a3b2d9aaf76dee678557f4a32d573c74f96c8122effa503",  # noqa: E501
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
