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
"""ImageClassifier Task presets"""

from keras_cv.models.backbones.resnet_v2 import resnet_v2_backbone_presets

classifier_presets = {
    "resnet50_v2_imagenet_classifier": {
        "metadata": {
            "description": (
                "ResNet classifier with 50 layers where the batch "
                "normalization and ReLU activation precede the convolution "
                "layers (v2 style). Trained on ILSVRC 2012 (Imagenet) "
                "classification task."
            ),
        },
        "config": {
            "backbone": resnet_v2_backbone_presets.backbone_presets[
                "resnet50_v2"
            ],
            "num_classes": 1000,
            "pooling": "avg",
        },
        # TODO(jbischof): fix checkpoint conversion for classification head
        "weights_url": "https://storage.googleapis.com/keras-cv/models/resnet50v2/imagenet/classification-v2.h5",
        "weights_hash": "5ee5a8ac650aaa59342bc48ffe770e6797a5550bcc35961e1d06685292c15921",
    },
}
