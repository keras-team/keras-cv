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
"""CLIP presets."""

clip_presets = {
    "clip-vit-base-patch16": {
        "metadata": {
            "description": (
                "The model uses a ViT-B/16 Transformer architecture as an "
                "image encoder and uses a masked self-attention Transformer as "
                "a text encoder. These encoders are trained to maximize the "
                "similarity of (image, text) pairs via a contrastive loss. The "
                "model uses a patch size of 16 and input images of size (224, "
                "224)"
            ),
            "params": 149620737,
            "official_name": "CLIP",
            "path": "clip",
        },
        "kaggle_handle": "kaggle://keras/clip/keras/clip-vit-base-patch16/7",
    },
    "clip-vit-base-patch32": {
        "metadata": {
            "description": (
                "The model uses a ViT-B/32 Transformer architecture as an "
                "image encoder and uses a masked self-attention Transformer as "
                "a text encoder. These encoders are trained to maximize the "
                "similarity of (image, text) pairs via a contrastive loss.The "
                "model uses a patch size of 32 and input images of size (224, "
                "224)"
            ),
            "params": 151277313,
            "official_name": "CLIP",
            "path": "clip",
        },
        "kaggle_handle": "kaggle://keras/clip/keras/clip-vit-base-patch32/6",
    },
    "clip-vit-large-patch14": {
        "metadata": {
            "description": (
                "The model uses a ViT-L/14 Transformer architecture as an "
                "image encoder and uses a masked self-attention Transformer as "
                "a text encoder. These encoders are trained to maximize the "
                "similarity of (image, text) pairs via a contrastive loss.The "
                "model uses a patch size of 14 and input images of size (224, "
                "224)"
            ),
            "params": 427616513,
            "official_name": "CLIP",
            "path": "clip",
        },
        "kaggle_handle": "kaggle://keras/clip/keras/clip-vit-large-patch14/6",
    },
    "clip-vit-large-patch14-336": {
        "metadata": {
            "description": (
                "The model uses a ViT-L/14 Transformer architecture as an "
                "image encoder and uses a masked self-attention Transformer as "
                "a text encoder. These encoders are trained to maximize the "
                "similarity of (image, text) pairs via a contrastive loss.The "
                "model uses a patch size of 14 and input images of size (336, "
                "336)"
            ),
            "params": 427944193,
            "official_name": "CLIP",
            "path": "clip",
        },
        "kaggle_handle": "kaggle://keras/clip/keras/clip-vit-large-patch14-336/6",  # noqa: E501
    },
}
