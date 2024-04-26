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

"""EfficientNetV1 model preset configurations."""

backbone_presets_no_weights = {
    "efficientnetv1_b0": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.0` and `depth_coefficient=1.0`."
            ),
            "params": 4050716,
            "official_name": "EfficientNetV1",
            "path": "efficientnetv1",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv1_b0",
    },
    "efficientnetv1_b1": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.0` and `depth_coefficient=1.1`."
            ),
            "params": 6576704,
            "official_name": "EfficientNetV1",
            "path": "efficientnetv1",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv1_b1",
    },
    "efficientnetv1_b2": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.1` and `depth_coefficient=1.2`."
            ),
            "params": 7770034,
            "official_name": "EfficientNetV1",
            "path": "efficientnetv1",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv1_b2",
    },
    "efficientnetv1_b3": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.2` and `depth_coefficient=1.4`."
            ),
            "params": 10785960,
            "official_name": "EfficientNetV1",
            "path": "efficientnetv1",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv1_b3",
    },
    "efficientnetv1_b4": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.4` and `depth_coefficient=1.8`."
            ),
            "params": 17676984,
            "official_name": "EfficientNetV1",
            "path": "efficientnetv1",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv1_b4",
    },
    "efficientnetv1_b5": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.6` and `depth_coefficient=2.2`."
            ),
            "params": 28517360,
            "official_name": "EfficientNetV1",
            "path": "efficientnetv1",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv1_b5",
    },
    "efficientnetv1_b6": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.8` and `depth_coefficient=2.6`."
            ),
            "params": 40965800,
            "official_name": "EfficientNetV1",
            "path": "efficientnetv1",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv1_b6",
    },
    "efficientnetv1_b7": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=2.0` and `depth_coefficient=3.1`."
            ),
            "params": 64105488,
            "official_name": "EfficientNetV1",
            "path": "efficientnetv1",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv1_b7",
    },
}

backbone_presets_with_weights = {}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
