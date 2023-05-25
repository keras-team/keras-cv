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
        "class_name": "keras_cv.models>DenseNetBackbone",
        "config": {
            "stackwise_num_repeats": [6, 12, 24, 16],
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "pooling": None,
            "compression_ratio": 0.5,
            "growth_rate": 32,
        },
    },
    "densenet169": {
        "metadata": {
            "description": "DenseNet model with 169 layers.",
        },
        "class_name": "keras_cv.models>DenseNetBackbone",
        "config": {
            "stackwise_num_repeats": [6, 12, 32, 32],
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "pooling": None,
            "compression_ratio": 0.5,
            "growth_rate": 32,
        },
    },
    "densenet201": {
        "metadata": {
            "description": "DenseNet model with 201 layers.",
        },
        "class_name": "keras_cv.models>DenseNetBackbone",
        "config": {
            "stackwise_num_repeats": [6, 12, 48, 32],
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "pooling": None,
            "compression_ratio": 0.5,
            "growth_rate": 32,
        },
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
        "class_name": "keras_cv.models>DenseNetBackbone",
        "config": backbone_presets_no_weights["densenet121"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/densenet121/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "709afe0321d9f2b2562e562ff9d0dc44cca10ed09e0e2cfba08d783ff4dab6bf",  # noqa: E501
    },
    "densenet169_imagenet": {
        "metadata": {
            "description": (
                "DenseNet model with 169 layers. Trained on Imagenet 2012 "
                "classification task."
            ),
        },
        "class_name": "keras_cv.models>DenseNetBackbone",
        "config": backbone_presets_no_weights["densenet169"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/densenet169/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "a99d1bb2cbe1a59a1cdd1f435fb265453a97c2a7b723d26f4ebee96e5fb49d62",  # noqa: E501
    },
    "densenet201_imagenet": {
        "metadata": {
            "description": (
                "DenseNet model with 201 layers. Trained on Imagenet 2012 "
                "classification task."
            ),
        },
        "class_name": "keras_cv.models>DenseNetBackbone",
        "config": backbone_presets_no_weights["densenet201"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/densenet201/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "c1189a934f12c1a676a9cf52238e5994401af925e2adfc0365bad8133c052060",  # noqa: E501
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
