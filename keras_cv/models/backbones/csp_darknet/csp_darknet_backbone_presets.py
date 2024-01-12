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

"""CSPDarkNet model preset configurations."""

backbone_presets_no_weights = {
    "csp_darknet_tiny": {
        "metadata": {
            "description": (
                "CSPDarkNet model with [48, 96, 192, 384] channels and "
                "[1, 3, 3, 1] depths where the batch normalization "
                "and SiLU activation are applied after the convolution layers."
            ),
            "params": 2380416,
            "official_name": "CSPDarkNet",
            "path": "csp_darknet",
        },
        "kaggle_handle": "kaggle://keras/cspdarknet/keras/csp_darknet_tiny/2",
    },
    "csp_darknet_s": {
        "metadata": {
            "description": (
                "CSPDarkNet model with [64, 128, 256, 512] channels and "
                "[1, 3, 3, 1] depths where the batch normalization "
                "and SiLU activation are applied after the convolution layers."
            ),
            "params": 4223488,
            "official_name": "CSPDarkNet",
            "path": "csp_darknet",
        },
        "kaggle_handle": "kaggle://keras/cspdarknet/keras/csp_darknet_s/2",
    },
    "csp_darknet_m": {
        "metadata": {
            "description": (
                "CSPDarkNet model with [96, 192, 384, 768] channels and "
                "[2, 6, 6, 2] depths where the batch normalization "
                "and SiLU activation are applied after the convolution layers."
            ),
            "params": 12374400,
            "official_name": "CSPDarkNet",
            "path": "csp_darknet",
        },
        "kaggle_handle": "kaggle://keras/cspdarknet/keras/csp_darknet_m/2",
    },
    "csp_darknet_l": {
        "metadata": {
            "description": (
                "CSPDarkNet model with [128, 256, 512, 1024] channels and "
                "[3, 9, 9, 3] depths where the batch normalization "
                "and SiLU activation are applied after the convolution layers."
            ),
            "params": 27111424,
            "official_name": "CSPDarkNet",
            "path": "csp_darknet",
        },
        "kaggle_handle": "kaggle://keras/cspdarknet/keras/csp_darknet_l/2",
    },
    "csp_darknet_xl": {
        "metadata": {
            "description": (
                "CSPDarkNet model with [170, 340, 680, 1360] channels and "
                "[4, 12, 12, 4] depths where the batch normalization "
                "and SiLU activation are applied after the convolution layers."
            ),
            "params": 56837970,
            "official_name": "CSPDarkNet",
            "path": "csp_darknet",
        },
        "kaggle_handle": "kaggle://keras/cspdarknet/keras/csp_darknet_xl/2",
    },
}

backbone_presets_with_weights = {
    "csp_darknet_tiny_imagenet": {
        "metadata": {
            "description": (
                "CSPDarkNet model with [48, 96, 192, 384] channels and "
                "[1, 3, 3, 1] depths where the batch normalization "
                "and SiLU activation are applied after the convolution layers. "
                "Trained on Imagenet 2012 classification task."
            ),
            "params": 2380416,
            "official_name": "CSPDarkNet",
            "path": "csp_darknet",
        },
        "kaggle_handle": "kaggle://keras/cspdarknet/keras/csp_darknet_tiny_imagenet/2",  # noqa: E501
    },
    "csp_darknet_l_imagenet": {
        "metadata": {
            "description": (
                "CSPDarkNet model with [128, 256, 512, 1024] channels and "
                "[3, 9, 9, 3] depths where the batch normalization "
                "and SiLU activation are applied after the convolution layers. "
                "Trained on Imagenet 2012 classification task."
            ),
            "params": 27111424,
            "official_name": "CSPDarkNet",
            "path": "csp_darknet",
        },
        "kaggle_handle": "kaggle://keras/cspdarknet/keras/csp_darknet_l_imagenet/2",  # noqa: E501
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
