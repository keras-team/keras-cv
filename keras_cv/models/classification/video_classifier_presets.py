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
"""VideoClassifier Task presets."""

classifier_presets = {
    "videoswin_tiny_kinetics_classifier": {
        "metadata": {
            "description": ("videoswin_tiny_kinetics "),  # TODO: update
            "params": 25_613_800,
            "official_name": "VideoClassifier",
            "path": "video_classifier",
        },
    },
    "videoswin_small_kinetics_classifier": {
        "metadata": {
            "description": ("videoswin_small_kinetics "),  # TODO: update
            "params": 25_613_800,  # TODO: update
            "official_name": "VideoClassifier",
            "path": "video_classifier",
        },
    },
    "videoswin_base_kinetics_classifier": {
        "metadata": {
            "description": ("videoswin_base_kinetics "),  # TODO: update
            "params": 25_613_800,  # TODO: update
            "official_name": "VideoClassifier",
            "path": "video_classifier",
        },
    },
    "videoswin_base_something_something_v2_classifier": {
        "metadata": {
            "description": (
                "videoswin_base_something_something_v2 "  # TODO: update
            ),
            "params": 25_613_800,  # TODO: update
            "official_name": "VideoClassifier",
            "path": "video_classifier",
        },
    },
}
