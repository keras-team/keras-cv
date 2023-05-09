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

yolo_v8_detector_presets = {
    "yolov8_m_pascalvoc": {
        "metadata": {
            "description": (
                "YOLOV8-M pretrained on PascalVOC 2012 object detection task, "
                "which consists of 20 classes. This model achieves a final MaP "
                "of 0.45 on the evaluation set."
            ),
            "params": 25901004,
            "official_name": "YOLOV8Detector",
            "path": "yolov8_detector",
        },
        "config": {
            "backbone": yolo_v8_backbone_presets.backbone_presets[
                "yolov8_m_backbone"
            ],
            "num_classes": 20,
            "fpn_depth": 2,
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/yolov8/pascal_voc/yolov8_m.h5",  # noqa: E501
        "weights_hash": "e641690aec205a3ca1ea730ea362ddc36c8b4a5abcebb6a23b18cbc9c091316d",  # noqa: E501
    },
}
