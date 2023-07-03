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
"""DeepLabV3+ Task Presets"""

from keras_cv.models import EfficientNetV2Backbone
from keras_cv.models import MobileNetV3Backbone
from keras_cv.models import ResNet50V2Backbone
from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_backbone_presets import (
    backbone_presets_no_weights as efficientnet_v2_backbone_presets,
)

deeplab_v3_plus_presets = {
    "deeplab_v3_plus_efficientnetv2_b0_cityscapes": {
        "metadata": {
            "description": (
                "DeepLabV3Plus with an EfficientNet V2 backbone with the "
                "preset `efficientnetv2_b0_imagenet`. Trained on Cityscapes "
                "pixel-level semantic labeling task, which consists of fine "
                "annotations for train and val sets (3475 annotated images) "
                "and 34 classes. This model achieves a final Mean IoU of "
                "0.36 on the validation set for fine annotations."
            ),
            "params": 15845136,
            "official_name": "DeepLabV3Plus",
            "path": "deeplab_v3_plus",
        },
        "config": {
            "backbone": efficientnet_v2_backbone_presets["efficientnetv2_b0_imagenet"],
            "num_classes": 34,
        },
        "weights_url": "https://huggingface.co/geekyrakshit/DeepLabV3-Plus/resolve/main/efficientnetv2_b0_imagenet.keras",
    },
}
