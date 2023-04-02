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
    "resnet50_pascal_voc": {
        "metadata": {
            "description": (
                "ResNet classifier with 50 layers where the batch "
                "normalization and ReLU activation precede the convolution "
                "layers (v2 style). Trained on Imagenet 2012 classification "
                "task."
            ),
        },
        "config": {
            "backbone": resnet_v1_backbone_presets.backbone_presets["resnet50"],
            # Should be 21
            "num_classes": 20,
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/retinanet/pascal_voc/resnet50.weights.h5",
        "weights_hash": "48d48d0bc53d2d135bf1c565b9c46efd",
    },
}
