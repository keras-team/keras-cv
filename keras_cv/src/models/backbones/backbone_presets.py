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

from keras_cv.src.models.backbones.csp_darknet import (
    csp_darknet_backbone_presets,
)
from keras_cv.src.models.backbones.densenet import densenet_backbone_presets
from keras_cv.src.models.backbones.efficientnet_lite import (
    efficientnet_lite_backbone_presets,
)
from keras_cv.src.models.backbones.efficientnet_v1 import (
    efficientnet_v1_backbone_presets,
)
from keras_cv.src.models.backbones.efficientnet_v2 import (
    efficientnet_v2_backbone_presets,
)
from keras_cv.src.models.backbones.mobilenet_v3 import (
    mobilenet_v3_backbone_presets,
)
from keras_cv.src.models.backbones.resnet_v1 import resnet_v1_backbone_presets
from keras_cv.src.models.backbones.resnet_v2 import resnet_v2_backbone_presets
from keras_cv.src.models.backbones.video_swin import video_swin_backbone_presets
from keras_cv.src.models.backbones.vit_det import vit_det_backbone_presets
from keras_cv.src.models.object_detection.yolo_v8 import (
    yolo_v8_backbone_presets,
)

backbone_presets_no_weights = {
    **resnet_v1_backbone_presets.backbone_presets_no_weights,
    **resnet_v2_backbone_presets.backbone_presets_no_weights,
    **mobilenet_v3_backbone_presets.backbone_presets_no_weights,
    **csp_darknet_backbone_presets.backbone_presets_no_weights,
    **efficientnet_v1_backbone_presets.backbone_presets_no_weights,
    **efficientnet_v2_backbone_presets.backbone_presets_no_weights,
    **densenet_backbone_presets.backbone_presets_no_weights,
    **efficientnet_lite_backbone_presets.backbone_presets_no_weights,
    **yolo_v8_backbone_presets.backbone_presets_no_weights,
    **vit_det_backbone_presets.backbone_presets_no_weights,
    **video_swin_backbone_presets.backbone_presets_no_weights,
}

backbone_presets_with_weights = {
    **resnet_v1_backbone_presets.backbone_presets_with_weights,
    **resnet_v2_backbone_presets.backbone_presets_with_weights,
    **mobilenet_v3_backbone_presets.backbone_presets_with_weights,
    **csp_darknet_backbone_presets.backbone_presets_with_weights,
    **efficientnet_v1_backbone_presets.backbone_presets_with_weights,
    **efficientnet_v2_backbone_presets.backbone_presets_with_weights,
    **densenet_backbone_presets.backbone_presets_with_weights,
    **efficientnet_lite_backbone_presets.backbone_presets_with_weights,
    **yolo_v8_backbone_presets.backbone_presets_with_weights,
    **vit_det_backbone_presets.backbone_presets_with_weights,
    **video_swin_backbone_presets.backbone_presets_with_weights,
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
