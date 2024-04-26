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
"""SAM model preset configurations."""


sam_presets = {
    "sam_base_sa1b": {
        "metadata": {
            "description": "The base SAM model trained on the SA1B dataset.",
            "params": 93_735_728,
            "official_name": "SAM",
            "path": "segment_anything",
        },
        "kaggle_handle": "kaggle://keras/sam/keras/sam_base_sa1b/2",
    },
    "sam_large_sa1b": {
        "metadata": {
            "description": "The large SAM model trained on the SA1B dataset.",
            "params": 312_343_088,
            "official_name": "SAM",
            "path": "segment_anything",
        },
        "kaggle_handle": "kaggle://keras/sam/keras/sam_large_sa1b/2",
    },
    "sam_huge_sa1b": {
        "metadata": {
            "description": "The huge SAM model trained on the SA1B dataset.",
            "params": 641_090_864,
            "official_name": "SAM",
            "path": "segment_anything",
        },
        "kaggle_handle": "kaggle://keras/sam/keras/sam_huge_sa1b/2",
    },
}
