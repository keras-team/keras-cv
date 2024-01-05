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
        "kaggle_handle": "kaggle://keras/mobilenetv3/keras/mobilenet_v3_small/2",  # noqa: E501
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
        "kaggle_handle": "kaggle://keras/mobilenetv3/keras/mobilenet_v3_large/2",  # noqa: E501
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
            "params": 2994518,
            "official_name": "MobileNetV3",
            "path": "mobilenetv3",
        },
        "kaggle_handle": "kaggle://keras/mobilenetv3/keras/mobilenet_v3_large_imagenet/2",  # noqa: E501
    },
    "mobilenet_v3_small_imagenet": {
        "metadata": {
            "description": (
                "MobileNetV3 model with 14 layers where the batch "
                "normalization and hard-swish activation are applied after the "
                "convolution layers. "
                "Pre-trained on the ImageNet 2012 classification task."
            ),
            "params": 933502,
            "official_name": "MobileNetV3",
            "path": "mobilenetv3",
        },
        "kaggle_handle": "kaggle://keras/mobilenetv3/keras/mobilenet_v3_small_imagenet/2",  # noqa: E501
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
