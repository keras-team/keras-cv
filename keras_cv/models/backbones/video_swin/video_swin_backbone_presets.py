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
"""Video Swin model preset configurations."""

backbone_presets_no_weights = {
    "videoswin_tiny": {
        "metadata": {
            "description": ("Video Swin backbone "),  # TODO: update
            "params": 27_850_470,
            "official_name": "VideoSwinT",
            "path": "video_swin",
        },
    },
    "videoswin_small": {
        "metadata": {
            "description": ("Video Swin backbone "),  # TODO: update
            "params": 49_509_078,
            "official_name": "VideoSwinS",
            "path": "video_swin",
        },
    },
    "videoswin_base": {
        "metadata": {
            "description": ("Video Swin backbone "),  # TODO: update
            "params": 87_638_984,
            "official_name": "VideoSwinB",
            "path": "video_swin",
        },
    },
}
