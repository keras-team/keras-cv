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
    "resnet18": {
        "metadata": {
            "description": (
                "ResNet model with 18 layers where the batch normalization "
                "and ReLU activation are applied after the convolution layers "
                "(v1 style)."
            ),
            "params": 11186112,
            "official_name": "ResNetV1",
            "path": "resnet_v1",
        },
        "kaggle_handle": "kaggle://keras/resnetv1/keras/resnet18/2",
    },
    "resnet34": {
        "metadata": {
            "description": (
                "ResNet model with 34 layers where the batch normalization "
                "and ReLU activation are applied after the convolution layers "
                "(v1 style)."
            ),
            "params": 21301696,
            "official_name": "ResNetV1",
            "path": "resnet_v1",
        },
        "kaggle_handle": "kaggle://keras/resnetv1/keras/resnet34/2",
    },
    "resnet50": {
        "metadata": {
            "description": (
                "ResNet model with 50 layers where the batch normalization "
                "and ReLU activation are applied after the convolution layers "
                "(v1 style)."
            ),
            "params": 23561152,
            "official_name": "ResNetV1",
            "path": "resnet_v1",
        },
        "kaggle_handle": "kaggle://keras/resnetv1/keras/resnet50/2",
    },
    "resnet101": {
        "metadata": {
            "description": (
                "ResNet model with 101 layers where the batch normalization "
                "and ReLU activation are applied after the convolution layers "
                "(v1 style)."
            ),
            "params": 42605504,
            "official_name": "ResNetV1",
            "path": "resnet_v1",
        },
        "kaggle_handle": "kaggle://keras/resnetv1/keras/resnet101/2",
    },
    "resnet152": {
        "metadata": {
            "description": (
                "ResNet model with 152 layers where the batch normalization "
                "and ReLU activation are applied after the convolution layers "
                "(v1 style)."
            ),
            "params": 58295232,
            "official_name": "ResNetV1",
            "path": "resnet_v1",
        },
        "kaggle_handle": "kaggle://keras/resnetv1/keras/resnet152/2",
    },
}

backbone_presets_with_weights = {
    "resnet50_imagenet": {
        "metadata": {
            "description": (
                "ResNet model with 50 layers where the batch normalization "
                "and ReLU activation are applied after the convolution layers "
                "(v1 style). "
                "Trained on Imagenet 2012 classification task."
            ),
            "params": 23561152,
            "official_name": "ResNetV1",
            "path": "resnet_v1",
        },
        "kaggle_handle": "kaggle://keras/resnetv1/keras/resnet50_imagenet/2",
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
