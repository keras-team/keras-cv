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
"""ImageClassifier Task presets."""

from keras_cv.models.backbones.efficientnet_v2 import (
    efficientnet_v2_backbone_presets,
)
from keras_cv.models.backbones.resnet_v2 import resnet_v2_backbone_presets

EFFICIENTNET_DESCRIPTION = "TODO"

classifier_presets = {
    "resnet50_v2_imagenet_classifier": {
        "metadata": {
            "description": (
                "ResNet classifier with 50 layers where the batch "
                "normalization and ReLU activation precede the convolution "
                "layers (v2 style). Trained on Imagenet 2012 classification "
                "task."
            ),
            "params": 25613800,
            "official_name": "ImageClassifier",
            "path": "image_classifier",
        },
        "config": {
            "backbone": resnet_v2_backbone_presets.backbone_presets[
                "resnet50_v2"
            ],
            "num_classes": 1000,
            "pooling": "avg",
            "activation": "softmax",
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/resnet50v2/imagenet-classifier-v0.h5",  # noqa: E501
        "weights_hash": "77fa9f1cd1de0e202309e51d4e598e441d1111dacb6c41a182b6c63f76ff26cd",  # noqa: E501
    },
    "efficientnetv2-s_imagenet_classifier": {
        "metadata": {
            "description": EFFICIENTNET_DESCRIPTION,
        },
        "config": {
            "backbone": efficientnet_v2_backbone_presets.backbone_presets[
                "efficientnetv2-s"
            ],
            "num_classes": 1000,
            "pooling": "avg",
            "activation": "softmax",
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/efficientnet_v2/efficientnetv2-s_imagenet_classifier.h5",
        "weights_hash": "4da57ade035e11aff7e1c5cb04c4235e",
    },
    "efficientnetv2-b0_imagenet_classifier": {
        "metadata": {
            "description": EFFICIENTNET_DESCRIPTION,
        },
        "config": {
            "backbone": efficientnet_v2_backbone_presets.backbone_presets[
                "efficientnetv2-b0"
            ],
            "num_classes": 1000,
            "pooling": "avg",
            "activation": "softmax",
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/efficientnet_v2/efficientnetv2-b0_imagenet_classifier.h5",
        "weights_hash": "4b739de648c346b7e4156e11a223c338",
    },
    "efficientnetv2-b1_imagenet_classifier": {
        "metadata": {
            "description": EFFICIENTNET_DESCRIPTION,
        },
        "config": {
            "backbone": efficientnet_v2_backbone_presets.backbone_presets[
                "efficientnetv2-b1"
            ],
            "num_classes": 1000,
            "pooling": "avg",
            "activation": "softmax",
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/efficientnet_v2/efficientnetv2-b1_imagenet_classifier.h5",
        "weights_hash": "78c1c879143dbd8f74e6ffc4d3180197",
    },
    "efficientnetv2-b2_imagenet_classifier": {
        "metadata": {
            "description": EFFICIENTNET_DESCRIPTION,
        },
        "config": {
            "backbone": efficientnet_v2_backbone_presets.backbone_presets[
                "efficientnetv2-b2"
            ],
            "num_classes": 1000,
            "pooling": "avg",
            "activation": "softmax",
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/efficientnet_v2/efficientnetv2-b2_imagenet_classifier.h5",
        "weights_hash": "07eda1c48aee27e12a3fe2545e6c65ed",
    },
}
