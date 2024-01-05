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

yolo_v8_detector_presets = {
    "yolo_v8_m_pascalvoc": {
        "metadata": {
            "description": (
                "YOLOV8-M pretrained on PascalVOC 2012 object detection task, "
                "which consists of 20 classes. This model achieves a final MaP "
                "of 0.45 on the evaluation set."
            ),
            "params": 25901004,
            "official_name": "YOLOV8Detector",
            "path": "yolo_v8_detector",
        },
        "kaggle_handle": "kaggle://keras/yolov8/keras/yolo_v8_m_pascalvoc/2",
    },
}
