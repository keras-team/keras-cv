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
"""YOLOv8 Backbone presets."""


backbone_presets = {
    "waymo_open_dataset": {
        "metadata": {
            "description": "An example CenterPillar backbone for WOD.",
            "params": 1277680,
            "official_name": "WaymoOpenDataset",
        },
        "class_name": (
            "keras_cv.models.object_detection_3d>CenterPillarBackbone"
        ),
        "config": {
            "stackwise_down_blocks": [6, 2, 1],
            "stackwise_down_filters": [128, 256, 512],
            "stackwise_up_filters": [512, 256, 256],
        },
    },
}
