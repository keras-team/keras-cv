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
"""VitDet model preset configurations."""

backbone_presets_no_weights = {
    "vitdet_base": {
        "metadata": {
            "description": (
                "Detectron2 ViT basebone with 12 "
                "transformer encoders with embed dim 768 and attention layers"
                " with 12 heads with global attention on encoders 2, 5, 8, "
                "and 11."
            ),
            "params": 89_670_912,
            "official_name": "VitDet",
            "path": "vit_det",
        },
        "kaggle_handle": "kaggle://keras/vitdet/keras/vitdet_base/2",
    },
    "vitdet_large": {
        "metadata": {
            "description": (
                "Detectron2 ViT basebone with 24 "
                "transformer encoders with embed dim "
                "1024 and attention layers with 16 heads with global "
                "attention on encoders 5, 11, 17, and 23."
            ),
            "params": 308_278_272,
            "official_name": "VitDet",
            "path": "vit_det",
        },
        "kaggle_handle": "kaggle://keras/vitdet/keras/vitdet_large/2",
    },
    "vitdet_huge": {
        "metadata": {
            "description": (
                "Detectron2 ViT basebone model "
                "with 32 transformer encoders with embed dim "
                "1280 and attention layers with 16 heads with global "
                "attention on encoders 7, 15, 23, and 31."
            ),
            "params": 637_026_048,
            "official_name": "VitDet",
            "path": "vit_det",
        },
        "kaggle_handle": "kaggle://keras/vitdet/keras/vitdet_huge/2",
    },
}


backbone_presets_with_weights = {
    "vitdet_base_sa1b": {
        "metadata": {
            "description": (
                "A base Detectron2 ViT backbone trained on the SA1B dataset."
            ),
            "params": 89_670_912,
            "official_name": "VitDet",
            "path": "vit_det",
        },
        "kaggle_handle": "kaggle://keras/vitdet/keras/vitdet_base_sa1b/2",
    },
    "vitdet_large_sa1b": {
        "metadata": {
            "description": (
                "A large Detectron2 ViT backbone trained on the SA1B dataset."
            ),
            "params": 308_278_272,
            "official_name": "VitDet",
            "path": "vit_det",
        },
        "kaggle_handle": "kaggle://keras/vitdet/keras/vitdet_large_sa1b/2",
    },
    "vitdet_huge_sa1b": {
        "metadata": {
            "description": (
                "A huge Detectron2 ViT backbone trained on the SA1B dataset."
            ),
            "params": 637_026_048,
            "official_name": "VitDet",
            "path": "vit_det",
        },
        "kaggle_handle": "kaggle://keras/vitdet/keras/vitdet_huge_sa1b/2",
    },
}


backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
