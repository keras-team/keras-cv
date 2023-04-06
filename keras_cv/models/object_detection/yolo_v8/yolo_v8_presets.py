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

# TODO(ianstenbit): This should preferably use CSPDarkNet presets eventually
yolo_v8_backbone_presets = {
    "yolov8_n_coco": {
        "metadata": {
            "description": ("YOLOv8_N backbone, pretrained on COCO dataset."),
        },
        "class_name": "keras_cv.models>YOLOV8Backbone",
        "config": {
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "channels": [32, 64, 128, 256],
            "depths": [1, 2, 2, 1],
        },
    },
}

yolo_v8_presets = {
    "yolov8_n_coco": {
        "metadata": {
            "description": ("TODO(ianstenbit): describe YOLOv8_N"),
        },
        "config": {
            "backbone": yolo_v8_backbone_presets["yolov8_n_coco"],
            "num_classes": 80,
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/yolov8/coco/yolov8_n.weights.h5",
        "weights_hash": "82d15d5a20cd2cf8bb2ab46cd3b7e9a9e0b32930844fcc4bd894099331a8e6fa",
    },
}
