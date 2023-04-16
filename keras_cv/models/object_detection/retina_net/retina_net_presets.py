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
"""RetinaNet Task presets."""
from keras_cv.models.backbones.resnet_v1 import resnet_v1_backbone_presets

retina_net_presets = {
    "retinanet_resnet50_pascalvoc": {
        "metadata": {
            "description": (
                "RetinaNet with a ResNet50 v1 backbone. "
                "Trained on PascalVOC 2012 object detection task, which "
                "consists of 20 classes. This model achieves a final MaP of "
                "0.33 on the evaluation set."
            ),
        },
        "config": {
            "backbone": resnet_v1_backbone_presets.backbone_presets["resnet50"],
            # 21 used as an implicit background class marginally improves
            # performance.
            "num_classes": 20,
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/retinanet/pascal_voc/resnet50-v3.weights.h5",  # noqa: E501
        "weights_hash": "84f51edc5d669109187b9c60edee1e55",
    },
}
