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
"""YOLOv8 Task presets."""

from keras_cv.models.object_detection.yolo_v8 import yolo_v8_backbone_presets

yolo_v8_presets = {
    "yolov8_n_coco": {
        "metadata": {
            "description": ("TODO(ianstenbit): describe YOLOv8_N"),
        },
        "config": {
            "backbone": yolo_v8_backbone_presets.backbone_presets[
                "yolov8_n_backbone"
            ],
            "num_classes": 80,
            "fpn_depth": 1,
            "temp_use_hacky_encoding": True,
        },
        "class_name": "keras_cv.models>YOLOV8",
        "weights_url": "https://storage.googleapis.com/keras-cv/models/yolov8/coco/yolov8_n.h5",
        "weights_hash": "2b96bd128a70a67a7226496319c1d4d8e33335e551b71b5a726bdc854b65e888",
    },
    "yolov8_s_coco": {
        "metadata": {
            "description": ("TODO(ianstenbit): describe YOLOv8_S"),
        },
        "config": {
            "backbone": yolo_v8_backbone_presets.backbone_presets[
                "yolov8_s_backbone"
            ],
            "num_classes": 80,
            "fpn_depth": 1,
            "temp_use_hacky_encoding": True,
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/yolov8/coco/yolov8_s.h5",
        "weights_hash": "3976460072d0b2d767540b4bafd07b4eaf12eeb4bcee9f53dab69968e8e7d95e",
    },
    "yolov8_m_coco": {
        "metadata": {
            "description": ("TODO(ianstenbit): describe YOLOv8_M"),
        },
        "config": {
            "backbone": yolo_v8_backbone_presets.backbone_presets[
                "yolov8_m_backbone"
            ],
            "num_classes": 80,
            "fpn_depth": 2,
            "temp_use_hacky_encoding": True,
        },
    },
    "yolov8_l_coco": {
        "metadata": {
            "description": ("TODO(ianstenbit): describe YOLOv8_L"),
        },
        "config": {
            "backbone": yolo_v8_backbone_presets.backbone_presets[
                "yolov8_l_backbone"
            ],
            "num_classes": 80,
            "fpn_depth": 3,
            "temp_use_hacky_encoding": True,
        },
    },
    "yolov8_x_coco": {
        "metadata": {
            "description": ("TODO(ianstenbit): describe YOLOv8_x"),
        },
        "config": {
            "backbone": yolo_v8_backbone_presets.backbone_presets[
                "yolov8_x_backbone"
            ],
            "num_classes": 80,
            "fpn_depth": 3,
            "temp_use_hacky_encoding": True,
        },
    },
    "yolov8_x6_coco": {
        "metadata": {
            "description": ("TODO(ianstenbit): describe YOLOv8_X6"),
        },
        "config": {
            "backbone": yolo_v8_backbone_presets.backbone_presets[
                "yolov8_x6_backbone"
            ],
            "num_classes": 80,
            "fpn_depth": 3,
            "temp_use_hacky_encoding": True,
        },
    },
}
