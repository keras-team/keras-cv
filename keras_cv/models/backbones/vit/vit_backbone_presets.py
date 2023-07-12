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
"""ViT model preset configurations"""

backbone_presets_no_weights = {
    "vittiny16": {
        "metadata": {
            "description": (
                "Vision Transformer Model with patch size of 16x16,"
                "3 heads, and a hidden size of 192."
            ),
        },
        "class_name": "keras_cv.models>ViTBackbone",
        "config": {
            "patch_size": 16,
            "transformer_layer_num": 12,
            "project_dim": 192,
            "mlp_dim": 768,
            "num_heads": 3,
            "mlp_dropout": 0.0,
            "attention_dropout": 0.0,
        },
    },
    "vits16": {
        "metadata": {
            "description": (
                "Vision Transformer Model with patch size of 16x16,"
                "6 heads, and a hidden size of 384."
            ),
        },
        "class_name": "keras_cv.models>ViTBackbone",
        "config": {
            "patch_size": 16,
            "transformer_layer_num": 12,
            "project_dim": 384,
            "mlp_dim": 1536,
            "num_heads": 6,
            "mlp_dropout": 0.0,
            "attention_dropout": 0.0,
        },
    },
    "vitb16": {
        "metadata": {
            "description": (
                "Vision Transformer Model with patch size of 16x16,"
                "12 heads, and a hidden size of 768."
            ),
        },
        "class_name": "keras_cv.models>ViTBackbone",
        "config": {
            "patch_size": 16,
            "transformer_layer_num": 12,
            "project_dim": 768,
            "mlp_dim": 3072,
            "num_heads": 12,
            "mlp_dropout": 0.0,
            "attention_dropout": 0.0,
        },
    },
    "vitl16": {
        "metadata": {
            "description": (
                "Vision Transformer Model with patch size of 16x16,"
                "16 heads, and a hidden size of 1024."
            ),
        },
        "class_name": "keras_cv.models>ViTBackbone",
        "config": {
            "patch_size": 16,
            "transformer_layer_num": 24,
            "project_dim": 1024,
            "mlp_dim": 4096,
            "num_heads": 16,
            "mlp_dropout": 0.1,
            "attention_dropout": 0.0,
        },
    },
    "vith16": {
        "metadata": {
            "description": (
                "Vision Transformer Model with patch size of 16x16,"
                "16 heads, and a hidden size of 1280."
            ),
        },
        "class_name": "keras_cv.models>ViTBackbone",
        "config": {
            "patch_size": 16,
            "transformer_layer_num": 32,
            "project_dim": 1280,
            "mlp_dim": 5120,
            "num_heads": 16,
            "mlp_dropout": 0.1,
            "attention_dropout": 0.0,
        },
    },
    "vittiny32": {
        "metadata": {
            "description": (
                "Vision Transformer Model with patch size of 32x32,"
                "3 heads, and a hidden size of 192."
            ),
        },
        "class_name": "keras_cv.models>ViTBackbone",
        "config": {
            "patch_size": 32,
            "transformer_layer_num": 12,
            "project_dim": 192,
            "mlp_dim": 768,
            "num_heads": 3,
            "mlp_dropout": 0.0,
            "attention_dropout": 0.0,
        },
    },
    "vits32": {
        "metadata": {
            "description": (
                "Vision Transformer Model with patch size of 32x32,"
                "6 heads, and a hidden size of 384."
            ),
        },
        "class_name": "keras_cv.models>ViTBackbone",
        "config": {
            "patch_size": 32,
            "transformer_layer_num": 12,
            "project_dim": 384,
            "mlp_dim": 1536,
            "num_heads": 6,
            "mlp_dropout": 0.0,
            "attention_dropout": 0.0,
        },
    },
    "vitb32": {
        "metadata": {
            "description": (
                "Vision Transformer Model with patch size of 32x32,"
                "12 heads, and a hidden size of 768."
            ),
        },
        "class_name": "keras_cv.models>ViTBackbone",
        "config": {
            "patch_size": 32,
            "transformer_layer_num": 12,
            "project_dim": 768,
            "mlp_dim": 3072,
            "num_heads": 12,
            "mlp_dropout": 0.0,
            "attention_dropout": 0.0,
        },
    },
    "vitl32": {
        "metadata": {
            "description": (
                "Vision Transformer Model with patch size of 32x32,"
                "16 heads, and a hidden size of 1024."
            ),
        },
        "class_name": "keras_cv.models>ViTBackbone",
        "config": {
            "patch_size": 32,
            "transformer_layer_num": 24,
            "project_dim": 1024,
            "mlp_dim": 4096,
            "num_heads": 16,
            "mlp_dropout": 0.1,
            "attention_dropout": 0.0,
        },
    },
    "vith32": {
        "metadata": {
            "description": (
                "Vision Transformer Model with patch size of 32x32,"
                "16 heads, and a hidden size of 1280."
            ),
        },
        "class_name": "keras_cv.models>ViTBackbone",
        "config": {
            "patch_size": 32,
            "transformer_layer_num": 32,
            "project_dim": 1280,
            "mlp_dim": 5120,
            "num_heads": 16,
            "mlp_dropout": 0.1,
            "attention_dropout": 0.0,
        },
    },
}

backbone_presets_with_weights = {
    "vittiny16_imagenet": {
        "metadata": {
            "description": (
                "Vision Transformer Model with patch size of 16x16,"
                "3 heads, and a hidden size of 192."
                "Pre-Trained on ImageNet (will only work with images "
                "of size (224, 224, 3).)"
                "Has an ImageNet top-1 accuracy of 78.22%."
            ),
        },
        "class_name": "keras_cv.models>ViTBackbone",
        "config": backbone_presets_no_weights["vittiny16"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/vittiny16/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "aa4d727e3c6bd30b20f49d3fa294fb4bbef97365c7dcb5cee9c527e4e83c8f5b",  # noqa: E501
    },
    "vits16_imagenet": {
        "metadata": {
            "description": (
                "Vision Transformer Model with patch size of 16x16,"
                "6 heads, and a hidden size of 384."
                "Pre-Trained on ImageNet (will only work with images of "
                "size (224, 224, 3).)"
                "Has an ImageNet top-1 accuracy of 83.73%."
            ),
        },
        "class_name": "keras_cv.models>ViTBackbone",
        "config": backbone_presets_no_weights["vits16"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/vits16/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "8d0111eda6692096676a5453abfec5d04c79e2de184b04627b295f10b1949745",  # noqa: E501
    },
    "vitb16_imagenet": {
        "metadata": {
            "description": (
                "Vision Transformer Model with patch size of 16x16,"
                "12 heads, and a hidden size of 768."
                "Pre-Trained on ImageNet (will only work with images of "
                "size (224, 224, 3).)"
                "Has an ImageNet top-1 accuracy of 85.49%."
            ),
        },
        "class_name": "keras_cv.models>ViTBackbone",
        "config": backbone_presets_no_weights["vitb16"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/vitb16/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "4a1bdd32889298471cb4f30882632e5744fd519bf1a1525b1fa312fe4ea775ed",  # noqa: E501
    },
    "vitl16_imagenet": {
        "metadata": {
            "description": (
                "Vision Transformer Model with patch size of 16x16,"
                "16 heads, and a hidden size of 1024."
                "Pre-Trained on ImageNet (will only work with images of "
                "size (224, 224, 3).)"
                "Has an ImageNet top-1 accuracy of 85.59%."
            ),
        },
        "class_name": "keras_cv.models>ViTBackbone",
        "config": backbone_presets_no_weights["vitl16"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/vitl16/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "40d237c44f14d20337266fce6192c00c2f9b890a463fd7f4cb17e8e35b3f5448",  # noqa: E501
    },
    "vits32_imagenet": {
        "metadata": {
            "description": (
                "Vision Transformer Model with patch size of 32x32,"
                "6 heads, and a hidden size of 384."
                "Pre-Trained on ImageNet (will only work with images of "
                "size (224, 224, 3).)"
                "Has an ImageNet top-1 accuracy of 79.58%."
            ),
        },
        "class_name": "keras_cv.models>ViTBackbone",
        "config": backbone_presets_no_weights["vits32"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/vits32/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "f3907845eff780a4d29c1c56e0ae053411f02fff6fdce1147c4c3bb2124698cd",  # noqa: E501
    },
    "vitb32_imagenet": {
        "metadata": {
            "description": (
                "Vision Transformer Model with patch size of 32x32,"
                "12 heads, and a hidden size of 768."
                "Pre-Trained on ImageNet (will only work with images of "
                "size (224, 224, 3).)"
                "Has an ImageNet top-1 accuracy of 83.59%."
            ),
        },
        "class_name": "keras_cv.models>ViTBackbone",
        "config": backbone_presets_no_weights["vitb32"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/vitb32/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "f07b80c03336d731a2a3a02af5cac1e9fc9aa62659cd29e2e7e5c7474150cc71",  # noqa: E501
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
