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
"""Densenet model preset configurations."""

backbone_presets_no_weights = {
    "DenseNet121": {
        "metadata": {
            "description": (
    
            ),
        },
        "class_name": "keras_cv.models>DenseNetBackbone",
        "config": {
            "blocks": [6, 12, 24, 16],
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "block_type": "basic_block",
        },
    },
    "DenseNet169": {
        "metadata": {
            "description": (
    
            ),
        },
        "class_name": "keras_cv.models>DenseNetBackbone",
        "config": {
            "blocks": [6, 12, 32, 32],
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "block_type": "basic_block",
        },
    },
    "DenseNet201": {
        "metadata": {
            "description": (
    
            ),
        },
        "class_name": "keras_cv.models>DenseNetBackbone",
        "config": {
            "blocks": [6, 12, 48, 32],
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "block_type": "basic_block",
        },
    },
}

backbone_presets_with_weights = {
    "DenseNet121_imagenet": {
        "metadata": {
            "description": (

            ),
        },
        "class_name": "keras_cv.models>DenseNetBackbone",
        "config": backbone_presets_no_weights["DenseNet121"]["config"],
        "weights_url": "13de3d077ad9d9816b9a0acc78215201d9b6e216c7ed8e71d69cc914f8f0775b",  # noqa: E501
        "weights_hash": "709afe0321d9f2b2562e562ff9d0dc44cca10ed09e0e2cfba08d783ff4dab6bf",  # noqa: E501
    },
}


backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
