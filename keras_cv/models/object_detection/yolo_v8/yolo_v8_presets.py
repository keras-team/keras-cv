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
yolo_v8_backbone_presets_no_weights = {
    "yolov8_n": {
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
    "yolov8_s": {
        "metadata": {
            "description": ("YOLOv8_S backbone, pretrained on COCO dataset."),
        },
        "class_name": "keras_cv.models>YOLOV8Backbone",
        "config": {
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "channels": [64, 128, 256, 512],
            "depths": [1, 2, 2, 1],
        },
    },
    "yolov8_m": {
        "metadata": {
            "description": ("YOLOv8_M backbone, pretrained on COCO dataset."),
        },
        "class_name": "keras_cv.models>YOLOV8Backbone",
        "config": {
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "channels": [96, 192, 384, 768],
            "depths": [2, 4, 4, 2],
        },
    },
    "yolov8_l": {
        "metadata": {
            "description": ("YOLOv8_L backbone, pretrained on COCO dataset."),
        },
        "class_name": "keras_cv.models>YOLOV8Backbone",
        "config": {
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "channels": [128, 256, 512, 512],
            "depths": [3, 6, 6, 3],
        },
    },
    "yolov8_x": {
        "metadata": {
            "description": ("YOLOv8_X backbone, pretrained on COCO dataset."),
        },
        "class_name": "keras_cv.models>YOLOV8Backbone",
        "config": {
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "channels": [160, 320, 640, 640],
            "depths": [3, 6, 6, 3],
        },
    },
    "yolov8_x6": {
        "metadata": {
            "description": ("YOLOv8_X6 backbone, pretrained on COCO dataset."),
        },
        "class_name": "keras_cv.models>YOLOV8Backbone",
        "config": {
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "channels": [160, 320, 640, 640, 640],
            "depths": [3, 6, 6, 3, 3],
        },
    },
}

yolo_v8_backbone_presets_with_weights = {
    "yolov8_n_coco": {
        "metadata": {"description": ("TODO: ianstenbit")},
        "class_name": "keras_cv.models>YOLOV8Backbone",
        "config": yolo_v8_backbone_presets_no_weights["yolov8_n"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/yolov8/coco/yolov8_n_backbone.h5",
        "weights_hash": "636ba3cba064c7c457e30a0e4759716006c305c30876df1c1caf2e56b99eab6c",
    },
}

yolo_v8_backbone_presets = {
    **yolo_v8_backbone_presets_no_weights,
    **yolo_v8_backbone_presets_with_weights,
}

yolo_v8_presets = {
    "yolov8_n_coco": {
        "metadata": {
            "description": ("TODO(ianstenbit): describe YOLOv8_N"),
        },
        "config": {
            "backbone": yolo_v8_backbone_presets["yolov8_n"],
            "num_classes": 80,
            "fpn_depth": 1,
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/yolov8/coco/yolov8_n.h5",
        "weights_hash": "82d15d5a20cd2cf8bb2ab46cd3b7e9a9e0b32930844fcc4bd894099331a8e6fa",
    },
    "yolov8_s_coco": {
        "metadata": {
            "description": ("TODO(ianstenbit): describe YOLOv8_S"),
        },
        "config": {
            "backbone": yolo_v8_backbone_presets["yolov8_s"],
            "num_classes": 80,
            "fpn_depth": 1,
        },
    },
    "yolov8_m_coco": {
        "metadata": {
            "description": ("TODO(ianstenbit): describe YOLOv8_M"),
        },
        "config": {
            "backbone": yolo_v8_backbone_presets["yolov8_m"],
            "num_classes": 80,
            "fpn_depth": 2,
        },
    },
    "yolov8_l_coco": {
        "metadata": {
            "description": ("TODO(ianstenbit): describe YOLOv8_L"),
        },
        "config": {
            "backbone": yolo_v8_backbone_presets["yolov8_l"],
            "num_classes": 80,
            "fpn_depth": 3,
        },
    },
    "yolov8_x_coco": {
        "metadata": {
            "description": ("TODO(ianstenbit): describe YOLOv8_x"),
        },
        "config": {
            "backbone": yolo_v8_backbone_presets["yolov8_x"],
            "num_classes": 80,
            "fpn_depth": 3,
        },
    },
    "yolov8_x6_coco": {
        "metadata": {
            "description": ("TODO(ianstenbit): describe YOLOv8_X6"),
        },
        "config": {
            "backbone": yolo_v8_backbone_presets["yolov8_x6"],
            "num_classes": 80,
            "fpn_depth": 3,
        },
    },
}
