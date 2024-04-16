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

from keras_cv.models.backbones.vit_det.vit_det_backbone_presets import (
    backbone_presets_no_weights,
)
from keras_cv.models.backbones.vit_det.vit_det_backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.models.backbones.vit_det.vit_det_backbone import (
    ViTDetBackbone,
)
from keras_cv.models.backbones.vit_det.vit_det_aliases import (
    ViTDetBBackbone, ViTDetLBackbone, ViTDetHBackbone,
)
from keras_cv.utils.preset_utils import register_presets, register_preset

register_presets(backbone_presets_no_weights, (ViTDetBackbone, ), with_weights=False)
register_presets(backbone_presets_with_weights, (ViTDetBackbone, ), with_weights=True)
register_preset("vitdet_base_sa1b", backbone_presets_with_weights["vitdet_base_sa1b"],
                (ViTDetBBackbone,), with_weights=True)
register_preset("vitdet_large_sa1b", backbone_presets_with_weights["vitdet_large_sa1b"],
                (ViTDetLBackbone,), with_weights=True)
register_preset("vitdet_huge_sa1b", backbone_presets_with_weights["vitdet_huge_sa1b"],
                (ViTDetHBackbone,), with_weights=True)
