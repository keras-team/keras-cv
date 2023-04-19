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

from keras_cv.models.backbones.resnet_v2 import resnet_v2_backbone_presets
from keras_cv.models.backbones.efficientnet_v2 import efficientnet_v2_backbone_presets

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
                "efficientnetv2-s_imagenet"
            ],
            "num_classes": 1000,
            "pooling": "avg",
            "activation": "softmax",
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/efficientnetv2s/imagenet/classification-v0.h5",
        "weights_hash": "2259db3483a577b5473dd406d1278439bd1a704ee477ff01a118299b134bd4db",
    },
    "efficientnetv2-b0_imagenet_classifier": {
        "metadata": {
            "description": EFFICIENTNET_DESCRIPTION,
        },
        "config": {
            "backbone": efficientnet_v2_backbone_presets.backbone_presets[
                "efficientnetv2-b0_imagenet"
            ],
            "num_classes": 1000,
            "pooling": "avg",
            "activation": "softmax",
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/efficientnetv2b0/imagenet/classification-v0.h5",
        "weights_hash": "dbde38e7c56af5bdafe61fd798cf5d490f3c5e3b699da7e25522bc828d208984",
    },
    "efficientnetv2-b1_imagenet_classifier": {
        "metadata": {
            "description": EFFICIENTNET_DESCRIPTION,
        },
        "config": {
            "backbone": efficientnet_v2_backbone_presets.backbone_presets[
                "efficientnetv2-b1_imagenet"
            ],
            "num_classes": 1000,
            "pooling": "avg",
            "activation": "softmax",
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/efficientnetv2b1/imagenet/classification-v0.h5",
        "weights_hash": "9dd8f3c8de3bbcc269a1b9aed742bb89d56be445b6aa271aa6037644f4210e9a",
    },
    "efficientnetv2-b2_imagenet_classifier": {
        "metadata": {
            "description": EFFICIENTNET_DESCRIPTION,
        },
        "config": {
            "backbone": efficientnet_v2_backbone_presets.backbone_presets[
                "efficientnetv2-b2_imagenet"
            ],
            "num_classes": 1000,
            "pooling": "avg",
            "activation": "softmax",
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/efficientnetv2b2/imagenet/classification-v0.h5",
        "weights_hash": "05eb5674e0ecbf34d5471f611bcfa5da0bb178332dc4460c7a911d68f9a2fe87",
    },
}
