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

"""All Backbone presets"""

from keras_cv.models.backbones.csp_darknet import csp_darknet_backbone_presets
from keras_cv.models.backbones.mobilenet_v3 import mobilenet_v3_backbone_presets
from keras_cv.models.backbones.resnet_v1 import resnet_v1_backbone_presets
from keras_cv.models.backbones.resnet_v2 import resnet_v2_backbone_presets

backbone_presets_no_weights = {
    **resnet_v1_backbone_presets.backbone_presets_no_weights,
    **resnet_v2_backbone_presets.backbone_presets_no_weights,
    **mobilenet_v3_backbone_presets.backbone_presets_no_weights,
    **csp_darknet_backbone_presets.backbone_presets_no_weights,
}

backbone_presets_with_weights = {
    **resnet_v1_backbone_presets.backbone_presets_with_weights,
    **resnet_v2_backbone_presets.backbone_presets_with_weights,
    **mobilenet_v3_backbone_presets.backbone_presets_with_weights,
    **csp_darknet_backbone_presets.backbone_presets_with_weights,
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
