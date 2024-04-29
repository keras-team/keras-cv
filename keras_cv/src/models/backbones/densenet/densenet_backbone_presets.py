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
"""DenseNet model preset configurations."""

backbone_presets_no_weights = {
    "densenet121": {
        "metadata": {
            "description": "DenseNet model with 121 layers.",
        },
        "kaggle_handle": "kaggle://keras/densenet/keras/densenet121/2",
    },
    "densenet169": {
        "metadata": {
            "description": "DenseNet model with 169 layers.",
        },
        "kaggle_handle": "kaggle://keras/densenet/keras/densenet169/2",
    },
    "densenet201": {
        "metadata": {
            "description": "DenseNet model with 201 layers.",
        },
        "kaggle_handle": "kaggle://keras/densenet/keras/densenet201/2",
    },
}

backbone_presets_with_weights = {
    "densenet121_imagenet": {
        "metadata": {
            "description": (
                "DenseNet model with 121 layers. Trained on Imagenet 2012 "
                "classification task."
            ),
        },
        "kaggle_handle": "kaggle://keras/densenet/keras/densenet121_imagenet/2",
    },
    "densenet169_imagenet": {
        "metadata": {
            "description": (
                "DenseNet model with 169 layers. Trained on Imagenet 2012 "
                "classification task."
            ),
        },
        "kaggle_handle": "kaggle://keras/densenet/keras/densenet169_imagenet/2",
    },
    "densenet201_imagenet": {
        "metadata": {
            "description": (
                "DenseNet model with 201 layers. Trained on Imagenet 2012 "
                "classification task."
            ),
        },
        "kaggle_handle": "kaggle://keras/densenet/keras/densenet201_imagenet/2",
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
