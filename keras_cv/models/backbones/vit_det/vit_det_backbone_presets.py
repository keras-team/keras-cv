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
        "class_name": "keras_cv.models>ViTDetBackbone",
        "config": {
            "input_shape": (1024, 1024, 3),
            "input_tensor": None,
            "include_rescaling": False,
            "patch_size": 16,
            "embed_dim": 768,
            "depth": 12,
            "mlp_dim": 768 * 4,
            "num_heads": 12,
            "out_chans": 256,
            "use_bias": True,
            "use_abs_pos": True,
            "use_rel_pos": True,
            "window_size": 14,
            "global_attention_indices": [2, 5, 8, 11],
            "layer_norm_epsilon": 1e-6,
        },
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
        "class_name": "keras_cv.models>ViTDetBackbone",
        "config": {
            "input_shape": (1024, 1024, 3),
            "input_tensor": None,
            "include_rescaling": False,
            "patch_size": 16,
            "embed_dim": 1024,
            "depth": 24,
            "mlp_dim": 1024 * 4,
            "num_heads": 16,
            "out_chans": 256,
            "use_bias": True,
            "use_abs_pos": True,
            "use_rel_pos": True,
            "window_size": 14,
            "global_attention_indices": [5, 11, 17, 23],
            "layer_norm_epsilon": 1e-6,
        },
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
        "class_name": "keras_cv.models>ViTDetBackbone",
        "config": {
            "input_shape": (1024, 1024, 3),
            "input_tensor": None,
            "include_rescaling": False,
            "patch_size": 16,
            "embed_dim": 1280,
            "depth": 32,
            "mlp_dim": 1280 * 4,
            "num_heads": 16,
            "out_chans": 256,
            "use_bias": True,
            "use_abs_pos": True,
            "use_rel_pos": True,
            "window_size": 14,
            "global_attention_indices": [7, 15, 23, 31],
            "layer_norm_epsilon": 1e-6,
        },
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
        "class_name": "keras_cv.models>ViTDetBackbone",
        "config": backbone_presets_no_weights["vitdet_base"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/vitdet/vitdet_base.h5",  # noqa: E501
        "weights_hash": "63c0ca6ff422142f95c24a0223445906728b353469be10c8e34018392207c93a",  # noqa: E501
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
        "class_name": "keras_cv.models>ViTDetBackbone",
        "config": backbone_presets_no_weights["vitdet_large"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/vitdet/vitdet_large.h5",  # noqa: E501
        "weights_hash": "b85f73ee5a82842aecbc7c706ca69530aaa828d3324d0793a93730c94727b30e",  # noqa: E501
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
        "class_name": "keras_cv.models>ViTDetBackbone",
        "config": backbone_presets_no_weights["vitdet_huge"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/vitdet/vitdet_huge.h5",  # noqa: E501
        "weights_hash": "ae6e1a95acd748f783bddeadd5915fdc6d1c15d23909df3cd4fa446c7d6b9fc1",  # noqa: E501
    },
}


backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
