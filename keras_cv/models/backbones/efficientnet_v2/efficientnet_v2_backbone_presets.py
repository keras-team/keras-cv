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

"""EfficientNetV2 model preset configurations."""

backbone_presets_no_weights = {
    "efficientnetv2_s": {
        "metadata": {
            "description": (
                "EfficientNet architecture with 6 convolutional blocks."
            ),
            "params": 20331360,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv2_s",
    },
    "efficientnetv2_m": {
        "metadata": {
            "description": (
                "EfficientNet architecture with 7 convolutional blocks."
            ),
            "params": 53150388,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv2_m",
    },
    "efficientnetv2_l": {
        "metadata": {
            "description": (
                "EfficientNet architecture with 7 convolutional "
                "blocks, but more filters the in `efficientnetv2_m`."
            ),
            "params": 117746848,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv2_l",
    },
    "efficientnetv2_b0": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 6 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.0` and `depth_coefficient=1.0`."
            ),
            "params": 5919312,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv2_b0",
    },
    "efficientnetv2_b1": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 6 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.0` and `depth_coefficient=1.1`."
            ),
            "params": 6931124,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv2_b1",
    },
    "efficientnetv2_b2": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 6 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.1` and `depth_coefficient=1.2`."
            ),
            "params": 8769374,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv2_b2",
    },
    "efficientnetv2_b3": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 7 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.2` and `depth_coefficient=1.4`."
            ),
            "params": 12930622,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv2_b3",
    },
}

backbone_presets_with_weights = {
    "efficientnetv2_s_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet architecture with 6 convolutional "
                "blocks. Weights are initialized to pretrained imagenet "
                "classification weights.Published weights are capable of "
                "scoring 83.9%top 1 accuracy "
                "and 96.7% top 5 accuracy on imagenet."
            ),
            "params": 20331360,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv2_s_imagenet",
    },
    "efficientnetv2_b0_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 6 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.0` and `depth_coefficient=1.0`. "
                "Weights are "
                "initialized to pretrained imagenet classification weights. "
                "Published weights are capable of scoring 77.1%	top 1 accuracy "
                "and 93.3% top 5 accuracy on imagenet."
            ),
            "params": 5919312,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv2_b0_imagenet",
    },
    "efficientnetv2_b1_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 6 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.0` and `depth_coefficient=1.1`. "
                "Weights are "
                "initialized to pretrained imagenet classification weights."
                "Published weights are capable of scoring 79.1%	top 1 accuracy "
                "and 94.4% top 5 accuracy on imagenet."
            ),
            "params": 6931124,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv2_b1_imagenet",
    },
    "efficientnetv2_b2_imagenet": {
        "metadata": {
            "description": (
                "EfficientNet B-style architecture with 6 "
                "convolutional blocks. This B-style model has "
                "`width_coefficient=1.1` and `depth_coefficient=1.2`. "
                "Weights are initialized to pretrained "
                "imagenet classification weights."
                "Published weights are capable of scoring 80.1%	top 1 accuracy "
                "and 94.9% top 5 accuracy on imagenet."
            ),
            "params": 8769374,
            "official_name": "EfficientNetV2",
            "path": "efficientnetv2",
        },
        "kaggle_handle": "gs://keras-cv-kaggle/efficientnetv2_b2_imagenet",
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
