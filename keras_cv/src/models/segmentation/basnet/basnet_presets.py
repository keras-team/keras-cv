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
"""BASNet model preset configurations."""

from keras_cv.src.models.backbones.resnet_v1 import resnet_v1_backbone_presets

presets_no_weights = {
    "basnet_resnet18": {
        "metadata": {
            "description": "BASNet with a ResNet18 v1 backbone.",
            "params": 98780872,
            "official_name": "BASNet",
            "path": "basnet_resnet18",
        },
        "config": {
            "backbone": resnet_v1_backbone_presets.backbone_presets["resnet18"],
            "num_classes": 1,
            "input_shape": (288, 288, 3),
        },
    },
    "basnet_resnet34": {
        "metadata": {
            "description": "BASNet with a ResNet34 v1 backbone.",
            "params": 108896456,
            "official_name": "BASNet",
            "path": "basnet_resnet34",
        },
        "config": {
            "backbone": resnet_v1_backbone_presets.backbone_presets["resnet34"],
            "num_classes": 1,
            "input_shape": (288, 288, 3),
        },
    },
}

presets_with_weights = {
    # TODO: Add BASNet preset with weights
}

basnet_presets = {**presets_no_weights, **presets_with_weights}
