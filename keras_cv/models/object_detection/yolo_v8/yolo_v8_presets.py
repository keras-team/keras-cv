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
            "description": ("An extra small YOLOV8 model pretrained on COCO"),
        },
        "config": {
            "backbone": yolo_v8_backbone_presets.backbone_presets[
                "yolov8_n_backbone"
            ],
            "num_classes": 80,
            "fpn_depth": 1,
        },
        "class_name": "keras_cv.models>YOLOV8",
        "weights_url": "https://storage.googleapis.com/keras-cv/models/yolov8/coco/yolov8_n.h5",  # noqa: E501
        "weights_hash": "2b96bd128a70a67a7226496319c1d4d8e33335e551b71b5a726bdc854b65e888",  # noqa: E501
    },
    "yolov8_s_coco": {
        "metadata": {
            "description": ("A small YOLOV8 model pretrained on COCO"),
        },
        "config": {
            "backbone": yolo_v8_backbone_presets.backbone_presets[
                "yolov8_s_backbone"
            ],
            "num_classes": 80,
            "fpn_depth": 1,
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/yolov8/coco/yolov8_s.h5",  # noqa: E501
        "weights_hash": "3976460072d0b2d767540b4bafd07b4eaf12eeb4bcee9f53dab69968e8e7d95e",  # noqa: E501
    },
    "yolov8_m_coco": {
        "metadata": {
            "description": ("A medium YOLOV8 model pretrained on COCO"),
        },
        "config": {
            "backbone": yolo_v8_backbone_presets.backbone_presets[
                "yolov8_m_backbone"
            ],
            "num_classes": 80,
            "fpn_depth": 2,
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/yolov8/coco/yolov8_m.h5",  # noqa: E501
        "weights_hash": "fb0909c45066a4d737ee170178476fbc8fe30226572c164422c59ebe2f0a79ce",  # noqa: E501
    },
    "yolov8_l_coco": {
        "metadata": {
            "description": ("A large YOLOV8 model pretrained on COCO"),
        },
        "config": {
            "backbone": yolo_v8_backbone_presets.backbone_presets[
                "yolov8_l_backbone"
            ],
            "num_classes": 80,
            "fpn_depth": 3,
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/yolov8/coco/yolov8_l.h5",  # noqa: E501
        "weights_hash": "7f424105a46b28d1f83be773acd85f50b0d4cf0b0fc5f266ad469e9ab7485d6e",  # noqa: E501
    },
    "yolov8_x_coco": {
        "metadata": {
            "description": ("An extra large YOLOV8 model pretrained on COCO"),
        },
        "config": {
            "backbone": yolo_v8_backbone_presets.backbone_presets[
                "yolov8_x_backbone"
            ],
            "num_classes": 80,
            "fpn_depth": 3,
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/yolov8/coco/yolov8_x.h5",  # noqa: E501
        "weights_hash": "f2916b94d872afd30b6893820cebdd00c745b271b1027523852001a99d62a2fa",  # noqa: E501
    },
}
