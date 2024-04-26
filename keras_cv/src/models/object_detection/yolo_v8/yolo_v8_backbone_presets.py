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


backbone_presets_no_weights = {
    "yolo_v8_xs_backbone": {
        "metadata": {
            "description": "An extra small YOLOV8 backbone",
            "params": 1277680,
            "official_name": "YOLOV8",
            "path": "yolo_v8",
        },
        "kaggle_handle": "kaggle://keras/yolov8/keras/yolo_v8_xs_backbone/2",
    },
    "yolo_v8_s_backbone": {
        "metadata": {
            "description": "A small YOLOV8 backbone",
            "params": 5089760,
            "official_name": "YOLOV8",
            "path": "yolo_v8",
        },
        "kaggle_handle": "kaggle://keras/yolov8/keras/yolo_v8_s_backbone/2",
    },
    "yolo_v8_m_backbone": {
        "metadata": {
            "description": "A medium YOLOV8 backbone",
            "params": 11872464,
            "official_name": "YOLOV8",
            "path": "yolo_v8",
        },
        "kaggle_handle": "kaggle://keras/yolov8/keras/yolo_v8_m_backbone/2",
    },
    "yolo_v8_l_backbone": {
        "metadata": {
            "description": "A large YOLOV8 backbone",
            "params": 19831744,
            "official_name": "YOLOV8",
            "path": "yolo_v8",
        },
        "kaggle_handle": "kaggle://keras/yolov8/keras/yolo_v8_l_backbone/2",
    },
    "yolo_v8_xl_backbone": {
        "metadata": {
            "description": "An extra large YOLOV8 backbone",
            "params": 30972080,
            "official_name": "YOLOV8",
            "path": "yolo_v8",
        },
        "kaggle_handle": "kaggle://keras/yolov8/keras/yolo_v8_xl_backbone/2",
    },
}

backbone_presets_with_weights = {
    "yolo_v8_xs_backbone_coco": {
        "metadata": {
            "description": (
                "An extra small YOLOV8 backbone pretrained on COCO"
            ),
            "params": 1277680,
            "official_name": "YOLOV8",
            "path": "yolo_v8",
        },
        "kaggle_handle": "kaggle://keras/yolov8/keras/yolo_v8_xs_backbone_coco/2",  # noqa: E501
    },
    "yolo_v8_s_backbone_coco": {
        "metadata": {
            "description": ("A small YOLOV8 backbone pretrained on COCO"),
            "params": 5089760,
            "official_name": "YOLOV8",
            "path": "yolo_v8",
        },
        "kaggle_handle": "kaggle://keras/yolov8/keras/yolo_v8_s_backbone_coco/2",  # noqa: E501
    },
    "yolo_v8_m_backbone_coco": {
        "metadata": {
            "description": ("A medium YOLOV8 backbone pretrained on COCO"),
            "params": 11872464,
            "official_name": "YOLOV8",
            "path": "yolo_v8",
        },
        "kaggle_handle": "kaggle://keras/yolov8/keras/yolo_v8_m_backbone_coco/2",  # noqa: E501
    },
    "yolo_v8_l_backbone_coco": {
        "metadata": {
            "description": ("A large YOLOV8 backbone pretrained on COCO"),
            "params": 19831744,
            "official_name": "YOLOV8",
            "path": "yolo_v8",
        },
        "kaggle_handle": "kaggle://keras/yolov8/keras/yolo_v8_l_backbone_coco/2",  # noqa: E501
    },
    "yolo_v8_xl_backbone_coco": {
        "metadata": {
            "description": (
                "An extra large YOLOV8 backbone pretrained on COCO"
            ),
            "params": 30972080,
            "official_name": "YOLOV8",
            "path": "yolo_v8",
        },
        "kaggle_handle": "kaggle://keras/yolov8/keras/yolo_v8_xl_backbone_coco/2",  # noqa: E501
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
