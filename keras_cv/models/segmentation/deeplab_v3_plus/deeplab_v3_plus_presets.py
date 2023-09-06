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
"""DeepLabV3Plus presets."""

from keras_cv.models.backbones.resnet_v2 import resnet_v2_backbone_presets

deeplab_v3_plus_presets = {
    "deeplab_v3_plus_resnet50_pascalvoc": {
        "metadata": {
            "description": (
                "DeeplabV3Plus with a ResNet50 v2 backbone. "
                "Trained on PascalVOC 2012 Semantic segmentation task, which "
                "consists of 20 classes and one background class. This model "
                "achieves a final categorical accuracy of 89.34% and mIoU of "
                "0.6391 on evaluation dataset."
            ),
            "params": 39191488,
            "official_name": "DeepLabV3Plus",
            "path": "deeplab_v3_plus",
        },
        "config": {
            "backbone": resnet_v2_backbone_presets.backbone_presets[
                "resnet50_v2"
            ],
            # 21 used as an implicit background class marginally improves
            # performance.
            "num_classes": 21,
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/deeplab_v3_plus/voc/deeplabv3plus_resenet50_pascal_voc.weights.h5",  # noqa: E501
        "weights_hash": "9681410a57bea2bc5cb7d79a1802d872ac263faab749cfe5ffdae6d6c3082041",  # noqa: E501
    },
}
