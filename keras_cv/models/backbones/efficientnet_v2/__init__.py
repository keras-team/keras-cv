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

from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_aliases import (
    EfficientNetV2B0Backbone,
)
from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_aliases import (
    EfficientNetV2B1Backbone,
)
from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_aliases import (
    EfficientNetV2B2Backbone,
)
from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_aliases import (
    EfficientNetV2SBackbone,
)
from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_backbone import (
    EfficientNetV2Backbone,
)
from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_backbone_presets import (
    backbone_presets_no_weights,
)
from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.utils.preset_utils import register_preset
from keras_cv.utils.preset_utils import register_presets

register_presets(
    backbone_presets_no_weights, (EfficientNetV2Backbone,), with_weights=False
)
register_presets(
    backbone_presets_with_weights, (EfficientNetV2Backbone,), with_weights=True
)
register_preset(
    "efficientnetv2_s_imagenet",
    backbone_presets_with_weights["efficientnetv2_s_imagenet"],
    (EfficientNetV2SBackbone,),
    with_weights=True,
)
register_preset(
    "efficientnetv2_b0_imagenet",
    backbone_presets_with_weights["efficientnetv2_b0_imagenet"],
    (EfficientNetV2B0Backbone,),
    with_weights=True,
)
register_preset(
    "efficientnetv2_b1_imagenet",
    backbone_presets_with_weights["efficientnetv2_b1_imagenet"],
    (EfficientNetV2B1Backbone,),
    with_weights=True,
)
register_preset(
    "efficientnetv2_b2_imagenet",
    backbone_presets_with_weights["efficientnetv2_b2_imagenet"],
    (EfficientNetV2B2Backbone,),
    with_weights=True,
)
