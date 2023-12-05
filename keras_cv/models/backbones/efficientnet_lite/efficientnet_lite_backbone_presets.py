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

"""EfficientNetLite model preset configurations."""

backbone_presets_no_weights = {
    "efficientnetlite_b0": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.0` and `depth_coefficient=1.0`."
            ),
            "params": 3414176,
            "official_name": "EfficientNetLite",
            "path": "EfficientNetLite",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetlite_b0",
    },
    "efficientnetlite_b1": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.0` and `depth_coefficient=1.1`."
            ),
            "params": 4190496,
            "official_name": "EfficientNetLite",
            "path": "EfficientNetLite",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetlite_b1",
    },
    "efficientnetlite_b2": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.1` and `depth_coefficient=1.2`."
            ),
            "params": 4870320,
            "official_name": "EfficientNetLite",
            "path": "EfficientNetLite",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetlite_b2",
    },
    "efficientnetlite_b3": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.2` and `depth_coefficient=1.4`."
            ),
            "params": 6994504,
            "official_name": "EfficientNetLite",
            "path": "EfficientNetLite",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetlite_b3",
    },
    "efficientnetlite_b4": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.4` and `depth_coefficient=1.8`."
            ),
            "params": 11840256,
            "official_name": "EfficientNetLite",
            "path": "EfficientNetLite",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetlite_b4",
    },
}

backbone_presets_with_weights = {}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
