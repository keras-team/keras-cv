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
"""MiT model preset configurations."""

backbone_presets_no_weights = {
    "mit_b0": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 8 transformer blocks."
            ),
            "params": 3321962,
            "official_name": "MiT",
            "path": "mit",
        },
        "kaggle_handle": "kaggle://keras/mit/keras/mit_b0/2",
    },
    "mit_b1": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 8 transformer blocks."
            ),
            "params": 13156554,
            "official_name": "MiT",
            "path": "mit",
        },
        "kaggle_handle": "kaggle://keras/mit/keras/mit_b1/2",
    },
    "mit_b2": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 16 transformer blocks."
            ),
            "params": 24201418,
            "official_name": "MiT",
            "path": "mit",
        },
        "kaggle_handle": "kaggle://keras/mit/keras/mit_b2/2",
    },
    "mit_b3": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 28 transformer blocks."
            ),
            "params": 44077258,
            "official_name": "MiT",
            "path": "mit",
        },
        "kaggle_handle": "kaggle://keras/mit/keras/mit_b3/2",
    },
    "mit_b4": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 41 transformer blocks."
            ),
            "params": 60847818,
            "official_name": "MiT",
            "path": "mit",
        },
        "kaggle_handle": "kaggle://keras/mit/keras/mit_b4/2",
    },
    "mit_b5": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 52 transformer blocks."
            ),
            "params": 81448138,
            "official_name": "MiT",
            "path": "mit",
        },
        "kaggle_handle": "kaggle://keras/mit/keras/mit_b5/2",
    },
}

backbone_presets_with_weights = {
    "mit_b0_imagenet": {
        "metadata": {
            "description": (
                "MiT (MixTransformer) model with 8 transformer blocks. Pre-trained on ImageNet-1K and scores 69% top-1 accuracy on the validation set."  # noqa: E501
            ),
            "params": 3321962,
            "official_name": "MiT",
            "path": "mit",
        },
        "kaggle_handle": "kaggle://keras/mit/keras/mit_b0_imagenet/2",
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
