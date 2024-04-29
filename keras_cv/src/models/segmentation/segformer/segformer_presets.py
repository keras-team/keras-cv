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
"""SegFormer model preset configurations."""

from keras_cv.src.models.backbones.mix_transformer.mix_transformer_backbone_presets import (  # noqa: E501
    backbone_presets,
)

presets_no_weights = {
    "segformer_b0": {
        "metadata": {
            "description": ("SegFormer model with MiTB0 backbone."),
            "params": 3719027,
            "official_name": "SegFormerB0",
            "path": "segformer_b0",
        },
        "kaggle_handle": "kaggle://keras/segformer/keras/segformer_b0/2",
    },
    "segformer_b1": {
        "metadata": {
            "description": ("SegFormer model with MiTB1 backbone."),
            "params": 13682643,
            "official_name": "SegFormerB1",
            "path": "segformer_b1",
        },
        "kaggle_handle": "kaggle://keras/segformer/keras/segformer_b1/2",
    },
    "segformer_b2": {
        "metadata": {
            "description": ("SegFormer model with MiTB2 backbone."),
            "params": 24727507,
            "official_name": "SegFormerB2",
            "path": "segformer_b2",
        },
        "kaggle_handle": "kaggle://keras/segformer/keras/segformer_b2/2",
    },
    "segformer_b3": {
        "metadata": {
            "description": ("SegFormer model with MiTB3 backbone."),
            "params": 44603347,
            "official_name": "SegFormerB3",
            "path": "segformer_b3",
        },
        "kaggle_handle": "kaggle://keras/segformer/keras/segformer_b3/2",
    },
    "segformer_b4": {
        "metadata": {
            "description": ("SegFormer model with MiTB4 backbone."),
            "params": 61373907,
            "official_name": "SegFormerB4",
            "path": "segformer_b4",
        },
        "kaggle_handle": "kaggle://keras/segformer/keras/segformer_b4/2",
    },
    "segformer_b5": {
        "metadata": {
            "description": ("SegFormer model with MiTB5 backbone."),
            "params": 81974227,
            "official_name": "SegFormerB5",
            "path": "segformer_b5",
        },
        "kaggle_handle": "kaggle://keras/segformer/keras/segformer_b5/2",
    },
}

presets_with_weights = {
    "segformer_b0_imagenet": {
        "metadata": {
            "description": (
                "SegFormer model with a pretrained MiTB0 backbone."
            ),
            "params": 3719027,
            "official_name": "SegFormerB0",
            "path": "segformer_b0",
        },
        "kaggle_handle": "kaggle://keras/segformer/keras/segformer_b0_imagenet/2",  # noqa: E501
    },
}

presets = {
    **backbone_presets,  # Add MiTBackbone presets
    **presets_no_weights,
    **presets_with_weights,
}
