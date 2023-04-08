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

MODEL_CONFIGS = {
    "DenseNet121": {
        "blocks": [6, 12, 24, 16],
    },
    "DenseNet169": {
        "blocks": [6, 12, 32, 32],
    },
    "DenseNet201": {
        "blocks": [6, 12, 48, 32],
    },
}
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
    "resnet50_imagenet": {
        "metadata": {
            "description": (

            ),
        },
        "class_name": "keras_cv.models>ResNetBackbone",
        "config": backbone_presets_no_weights["resnet50"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/resnet50/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "dc5f6d8f929c78d0fc192afecc67b11ac2166e9d8b9ef945742368ae254c07af",  # noqa: E501
    },
}


backbone_presets = {
    **backbone_presets_no_weights,
}
