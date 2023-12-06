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
    "center_pillar_waymo_open_dataset": {
        "metadata": {
            "description": "An example CenterPillar backbone for WOD.",
            "params": 1277680,
            "official_name": "CenterPillar",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/center_pillar_waymo_open_dataset",  # noqa: E501
    },
}
