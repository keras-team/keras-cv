# Copyright 2024 The KerasCV Authors
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

from keras_cv.models.backbones.video_swin.video_swin_aliases import (
    VideoSwinBBackbone,
)
from keras_cv.models.backbones.video_swin.video_swin_aliases import (
    VideoSwinSBackbone,
)
from keras_cv.models.backbones.video_swin.video_swin_aliases import (
    VideoSwinTBackbone,
)
from keras_cv.models.backbones.video_swin.video_swin_backbone import (
    VideoSwinBackbone,
)
from keras_cv.models.backbones.video_swin.video_swin_backbone_presets import (
    backbone_presets_no_weights,
)
from keras_cv.models.backbones.video_swin.video_swin_backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.utils.preset_utils import register_preset
from keras_cv.utils.preset_utils import register_presets

register_presets(
    backbone_presets_no_weights, (VideoSwinBackbone,), with_weights=False
)
register_presets(
    backbone_presets_with_weights, (VideoSwinBackbone,), with_weights=True
)
register_preset(
    "videoswin_tiny_kinetics400",
    backbone_presets_with_weights["videoswin_tiny_kinetics400"],
    (VideoSwinTBackbone,),
    with_weights=True,
)
register_preset(
    "videoswin_small_kinetics400",
    backbone_presets_with_weights["videoswin_small_kinetics400"],
    (VideoSwinSBackbone,),
    with_weights=True,
)
register_preset(
    "videoswin_base_kinetics400",
    backbone_presets_with_weights["videoswin_base_kinetics400"],
    (VideoSwinBBackbone,),
    with_weights=True,
)
