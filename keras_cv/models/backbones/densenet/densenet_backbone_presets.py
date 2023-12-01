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
        "kaggle_handle": "gs://keras-cv-kaggle/densenet121",
    },
    "densenet169": {
        "metadata": {
            "description": "DenseNet model with 169 layers.",
        },
        "class_name": "keras_cv>DenseNetBackbone",
        "config": {
            "stackwise_num_repeats": [6, 12, 32, 32],
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "compression_ratio": 0.5,
            "growth_rate": 32,
        },
        "kaggle_handle": "gs://keras-cv-kaggle/densenet169",
    },
    "densenet201": {
        "metadata": {
            "description": "DenseNet model with 201 layers.",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/densenet201",
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
        "kaggle_handle": "gs://keras-cv-kaggle/densenet121_imagenet",
    },
    "densenet169_imagenet": {
        "metadata": {
            "description": (
                "DenseNet model with 169 layers. Trained on Imagenet 2012 "
                "classification task."
            ),
        },
        "kaggle_handle": "gs://keras-cv-kaggle/densenet169_imagenet",
    },
    "densenet201_imagenet": {
        "metadata": {
            "description": (
                "DenseNet model with 201 layers. Trained on Imagenet 2012 "
                "classification task."
            ),
        },
        "kaggle_handle": "gs://keras-cv-kaggle/densenet201_imagenet",
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
