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
"""ResNetV2 model preset configurations."""

backbone_presets_no_weights = {
    "resnet18_v2": {
        "metadata": {
            "description": (
                "ResNet model with 18 layers where the batch normalization "
                "and ReLU activation precede the convolution layers (v2 style)."
            ),
            "params": 11183488,
            "official_name": "ResNetV2",
            "path": "resnet_v2",
        },
        "kaggle_handle": "kaggle://keras/resnetv2/keras/resnet18_v2/2",
    },
    "resnet34_v2": {
        "metadata": {
            "description": (
                "ResNet model with 34 layers where the batch normalization "
                "and ReLU activation precede the convolution layers (v2 style)."
            ),
            "params": 21299072,
            "official_name": "ResNetV2",
            "path": "resnet_v2",
        },
        "kaggle_handle": "kaggle://keras/resnetv2/keras/resnet34_v2/2",
    },
    "resnet50_v2": {
        "metadata": {
            "description": (
                "ResNet model with 50 layers where the batch normalization "
                "and ReLU activation precede the convolution layers (v2 style)."
            ),
            "params": 23564800,
            "official_name": "ResNetV2",
            "path": "resnet_v2",
        },
        "kaggle_handle": "kaggle://keras/resnetv2/keras/resnet50_v2/2",
    },
    "resnet101_v2": {
        "metadata": {
            "description": (
                "ResNet model with 101 layers where the batch normalization "
                "and ReLU activation precede the convolution layers (v2 style)."
            ),
            "params": 42626560,
            "official_name": "ResNetV2",
            "path": "resnet_v2",
        },
        "kaggle_handle": "kaggle://keras/resnetv2/keras/resnet101_v2/2",
    },
    "resnet152_v2": {
        "metadata": {
            "description": (
                "ResNet model with 152 layers where the batch normalization "
                "and ReLU activation precede the convolution layers (v2 style)."
            ),
            "params": 58331648,
            "official_name": "ResNetV2",
            "path": "resnet_v2",
        },
        "kaggle_handle": "kaggle://keras/resnetv2/keras/resnet152_v2/2",
    },
}

backbone_presets_with_weights = {
    "resnet50_v2_imagenet": {
        "metadata": {
            "description": (
                "ResNet model with 50 layers where the batch normalization and "
                "ReLU activation precede the convolution layers (v2 style). "
                "Trained on Imagenet 2012 classification task."
            ),
            "params": 23564800,
            "official_name": "ResNetV2",
            "path": "resnet_v2",
        },
        "kaggle_handle": "kaggle://keras/resnetv2/keras/resnet50_v2_imagenet/2",
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
