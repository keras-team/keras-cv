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
    "yolov8_xs_backbone": {
        "metadata": {
            "description": "An extra small YOLOV8 backbone",
            "params": 1277680,
            "official_name": "YOLOV8",
            "path": "yolov8",
        },
        "class_name": "keras_cv.models>YOLOV8Backbone",
        "config": {
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "channels": [32, 64, 128, 256],
            "depths": [1, 2, 2, 1],
            "activation": "swish",
        },
    },
    "yolov8_s_backbone": {
        "metadata": {
            "description": "A small YOLOV8 backbone",
            "params": 5089760,
            "official_name": "YOLOV8",
            "path": "yolov8",
        },
        "class_name": "keras_cv.models>YOLOV8Backbone",
        "config": {
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "channels": [64, 128, 256, 512],
            "depths": [1, 2, 2, 1],
            "activation": "swish",
        },
    },
    "yolov8_m_backbone": {
        "metadata": {
            "description": "A medium YOLOV8 backbone",
            "params": 11872464,
            "official_name": "YOLOV8",
            "path": "yolov8",
        },
        "class_name": "keras_cv.models>YOLOV8Backbone",
        "config": {
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "channels": [96, 192, 384, 576],
            "depths": [2, 4, 4, 2],
            "activation": "swish",
        },
    },
    "yolov8_l_backbone": {
        "metadata": {
            "description": "A large YOLOV8 backbone",
            "params": 19831744,
            "official_name": "YOLOV8",
            "path": "yolov8",
        },
        "class_name": "keras_cv.models>YOLOV8Backbone",
        "config": {
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "channels": [128, 256, 512, 512],
            "depths": [3, 6, 6, 3],
            "activation": "swish",
        },
    },
    "yolov8_xl_backbone": {
        "metadata": {
            "description": "An extra large YOLOV8 backbone",
            "params": 30972080,
            "official_name": "YOLOV8",
            "path": "yolov8",
        },
        "class_name": "keras_cv.models>YOLOV8Backbone",
        "config": {
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "channels": [160, 320, 640, 640],
            "depths": [3, 6, 6, 3],
            "activation": "swish",
        },
    },
}

backbone_presets_with_weights = {
    "yolov8_xs_backbone_coco": {
        "metadata": {
            "description": (
                "An extra small YOLOV8 backbone pretrained on COCO"
            ),
            "params": 1277680,
            "official_name": "YOLOV8",
            "path": "yolov8",
        },
        "class_name": "keras_cv.models>YOLOV8Backbone",
        "config": backbone_presets_no_weights["yolov8_xs_backbone"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/yolov8/coco/yolov8_n_backbone.h5",  # noqa: E501
        "weights_hash": "636ba3cba064c7c457e30a0e4759716006c305c30876df1c1caf2e56b99eab6c",  # noqa: E501
    },
    "yolov8_s_backbone_coco": {
        "metadata": {
            "description": ("A small YOLOV8 backbone pretrained on COCO"),
            "params": 5089760,
            "official_name": "YOLOV8",
            "path": "yolov8",
        },
        "class_name": "keras_cv.models>YOLOV8Backbone",
        "config": backbone_presets_no_weights["yolov8_s_backbone"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/yolov8/coco/yolov8_s_backbone.h5",  # noqa: E501
        "weights_hash": "49ab5da87d6b36a1943e7f111a1960355171332c25312b6cc01526baaecf1b69",  # noqa: E501
    },
    "yolov8_m_backbone_coco": {
        "metadata": {
            "description": ("A medium YOLOV8 backbone pretrained on COCO"),
            "params": 11872464,
            "official_name": "YOLOV8",
            "path": "yolov8",
        },
        "class_name": "keras_cv.models>YOLOV8Backbone",
        "config": backbone_presets_no_weights["yolov8_m_backbone"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/yolov8/coco/yolov8_m_backbone.h5",  # noqa: E501
        "weights_hash": "a9719807699a2540da14aa7f9a0dda272d400d30c40a956298a63a2805aa6436",  # noqa: E501
    },
    "yolov8_l_backbone_coco": {
        "metadata": {
            "description": ("A large YOLOV8 backbone pretrained on COCO"),
            "params": 19831744,
            "official_name": "YOLOV8",
            "path": "yolov8",
        },
        "class_name": "keras_cv.models>YOLOV8Backbone",
        "config": backbone_presets_no_weights["yolov8_l_backbone"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/yolov8/coco/yolov8_l_backbone.h5",  # noqa: E501
        "weights_hash": "2c94ffe75492491974c6d7347d5c1d1aa209d8f6d78c63ab62df0f5dd51680b9",  # noqa: E501
    },
    "yolov8_xl_backbone_coco": {
        "metadata": {
            "description": (
                "An extra large YOLOV8 backbone pretrained on COCO"
            ),
            "params": 30972080,
            "official_name": "YOLOV8",
            "path": "yolov8",
        },
        "class_name": "keras_cv.models>YOLOV8Backbone",
        "config": backbone_presets_no_weights["yolov8_xl_backbone"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/yolov8/coco/yolov8_x_backbone.h5",  # noqa: E501
        "weights_hash": "ce0cc3235eacaffc4a9824e28b2366e674b6d42befc4c7b77f3be7d1d39960bd",  # noqa: E501
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
