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

backbone_presets = {
    "sam_vitdet_b": {
        "metadata": {
            "description": (
                "VitDet Backbone for the segment anything model with 12 "
                "transformer encoders with embed dim 768 and attention layers"
                " with 12 heads with global attention on encoders 2, 5, 8, "
                "and 11."
            ),
            "params": 89_670_912,
            "official_name": "VitDet",
            "path": "detectron2",
        },
        "class_name": "keras_cv.models>VitDetBackbone",
        "config": {
            "img_size": 1024,
            "patch_size": 16,
            "in_chans": 3,
            "embed_dim": 768,
            "depth": 12,
            "mlp_dim": 768 * 4,
            "num_heads": 12,
            "out_chans": 256,
            "use_bias": True,
            "use_rel_pos": True,
            "window_size": 14,
            "global_attention_indices": [2, 5, 8, 11],
        },
        # "weights_url": "https://storage.googleapis.com/keras-cv/models/segment_anything/sam_vit_b.weights.h5",  # noqa: E501
        # "weights_hash": None
    },
    "sam_vitdet_l": {
        "metadata": {
            "description": (
                "VitDet Backbone for the segment anything model with 24 "
                "transformer encoders with embed dim "
                "1024 and attention layers with 16 heads with global "
                "attention on encoders 5, 11, 17, and 23."
            ),
            "params": 308_278_272,
            "official_name": "VitDet",
            "path": "detectron2",
        },
        "class_name": "keras_cv.models>VitDetBackbone",
        "config": {
            "img_size": 1024,
            "patch_size": 16,
            "in_chans": 3,
            "embed_dim": 1024,
            "depth": 24,
            "mlp_dim": 1024 * 4,
            "num_heads": 16,
            "out_chans": 256,
            "use_bias": True,
            "use_rel_pos": True,
            "window_size": 14,
            "global_attention_indices": [5, 11, 17, 23],
        },
        # "weights_url": "https://storage.googleapis.com/keras-cv/models/segment_anything/sam_vit_l.weights.h5",  # noqa: E501
        # "weights_hash": None
    },
    "sam_vitdet_h": {
        "metadata": {
            "description": (
                "VitDet Backbone  for the segment anything model "
                "with 32 transformer encoders with embed dim "
                "1280 and attention layers with 16 heads with global "
                "attention on encoders 7, 15, 23, and 31."
            ),
            "params": 637_026_048,
            "official_name": "VitDet",
            "path": "detectron2",
        },
        "class_name": "keras_cv.models>VitDetBackbone",
        "config": {
            "img_size": 1024,
            "patch_size": 16,
            "in_chans": 3,
            "embed_dim": 1280,
            "depth": 32,
            "mlp_dim": 1280 * 4,
            "num_heads": 16,
            "out_chans": 256,
            "use_bias": True,
            "use_rel_pos": True,
            "window_size": 14,
            "global_attention_indices": [7, 15, 23, 31],
        },
        # "weights_url": "https://storage.googleapis.com/keras-cv/models/segment_anything/sam_vit_h.weights.h5",  # noqa: E501
        # "weights_hash": None
    },
}
