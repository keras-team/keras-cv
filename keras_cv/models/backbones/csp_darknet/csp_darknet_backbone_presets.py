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

"""CSPDarkNet model preset configurations."""

backbone_presets_no_weights = {
    "csp_darknet_tiny": {
        "metadata": {
            "description": (
                "CSPDarkNet model with [48, 96, 192, 384] channels and "
                "[1, 3, 3, 1] depths where the batch normalization "
                "and SiLU activation are applied after the convolution layers."
            ),
            "params": 2380416,
            "official_name": "CSPDarkNet",
            "path": "csp_darknet",
        },
        "class_name": "keras_cv.models>CSPDarkNetBackbone",
        "config": {
            "stackwise_channels": [48, 96, 192, 384],
            "stackwise_depth": [1, 3, 3, 1],
            "include_rescaling": True,
            "use_depthwise": False,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "csp_darknet_s": {
        "metadata": {
            "description": (
                "CSPDarkNet model with [64, 128, 256, 512] channels and "
                "[1, 3, 3, 1] depths where the batch normalization "
                "and SiLU activation are applied after the convolution layers."
            ),
            "params": 4223488,
            "official_name": "CSPDarkNet",
            "path": "csp_darknet",
        },
        "class_name": "keras_cv.models>CSPDarkNetBackbone",
        "config": {
            "stackwise_channels": [64, 128, 256, 512],
            "stackwise_depth": [1, 3, 3, 1],
            "include_rescaling": True,
            "use_depthwise": False,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "csp_darknet_m": {
        "metadata": {
            "description": (
                "CSPDarkNet model with [96, 192, 384, 768] channels and "
                "[2, 6, 6, 2] depths where the batch normalization "
                "and SiLU activation are applied after the convolution layers."
            ),
            "params": 12374400,
            "official_name": "CSPDarkNet",
            "path": "csp_darknet",
        },
        "class_name": "keras_cv.models>CSPDarkNetBackbone",
        "config": {
            "stackwise_channels": [96, 192, 384, 768],
            "stackwise_depth": [2, 6, 6, 2],
            "include_rescaling": True,
            "use_depthwise": False,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "csp_darknet_l": {
        "metadata": {
            "description": (
                "CSPDarkNet model with [128, 256, 512, 1024] channels and "
                "[3, 9, 9, 3] depths where the batch normalization "
                "and SiLU activation are applied after the convolution layers."
            ),
            "params": 27111424,
            "official_name": "CSPDarkNet",
            "path": "csp_darknet",
        },
        "class_name": "keras_cv.models>CSPDarkNetBackbone",
        "config": {
            "stackwise_channels": [128, 256, 512, 1024],
            "stackwise_depth": [3, 9, 9, 3],
            "include_rescaling": True,
            "use_depthwise": False,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "csp_darknet_xl": {
        "metadata": {
            "description": (
                "CSPDarkNet model with [170, 340, 680, 1360] channels and "
                "[4, 12, 12, 4] depths where the batch normalization "
                "and SiLU activation are applied after the convolution layers."
            ),
            "params": 56837970,
            "official_name": "CSPDarkNet",
            "path": "csp_darknet",
        },
        "class_name": "keras_cv.models>CSPDarkNetBackbone",
        "config": {
            "stackwise_channels": [170, 340, 680, 1360],
            "stackwise_depth": [4, 12, 12, 4],
            "include_rescaling": True,
            "use_depthwise": False,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "yolov8_xl_backbone": {
        "metadata": {
            "description": "An extra large YOLOV8 backbone",
        },
        "class_name": "keras_cv.models>CSPDarkNetBackbone",
        "config": {
            "stackwise_channels": [160, 320, 640, 640],
            "stackwise_depth": [3, 6, 6, 3],
            "include_rescaling": True,
            "use_depthwise": False,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "use_focus": False,
            "batch_norm_momentum": 0.97,
            "stem_stride": 2,
            "spp_last": True,
            "use_zero_padding": True,
            "padding": "valid",
            "spp_pool_sizes": (5, 5, 5),
            "sequential_pooling": True,
            "wide_stem": True,
            "kernel_sizes": [3, 3],
            "concat_all": True,
            "always_residual": True,
        },
    },
}

backbone_presets_with_weights = {
    "csp_darknet_tiny_imagenet": {
        "metadata": {
            "description": (
                "CSPDarkNet model with [48, 96, 192, 384] channels and "
                "[1, 3, 3, 1] depths where the batch normalization "
                "and SiLU activation are applied after the convolution layers. "
                "Trained on Imagenet 2012 classification task."
            ),
            "params": 2380416,
            "official_name": "CSPDarkNet",
            "path": "csp_darknet",
        },
        "class_name": "keras_cv.models>CSPDarkNetBackbone",
        "config": backbone_presets_no_weights["csp_darknet_tiny"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/cspdarknettiny/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "0007ae82c95be4d4aef06368a7c38e006381324d77e5df029b04890e18a8ad19",  # noqa: E501
    },
    "csp_darknet_l_imagenet": {
        "metadata": {
            "description": (
                "CSPDarkNet model with [128, 256, 512, 1024] channels and "
                "[3, 9, 9, 3] depths where the batch normalization "
                "and SiLU activation are applied after the convolution layers. "
                "Trained on Imagenet 2012 classification task."
            ),
            "params": 27111424,
            "official_name": "CSPDarkNet",
            "path": "csp_darknet",
        },
        "class_name": "keras_cv.models>CSPDarkNetBackbone",
        "config": backbone_presets_no_weights["csp_darknet_l"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/cspdarknetl/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "9303aabfadffbff8447171fce1e941f96d230d8f3cef30d3f05a9c85097f8f1e",  # noqa: E501
    },
    "yolov8_xl_backbone_coco": {
        "metadata": {
            "description": ("An extra large YOLOV8 backbone pretrained on COCO")
        },
        "class_name": "keras_cv.models>CSPDarkNetBackbone",
        "config": backbone_presets_no_weights["yolov8_xl_backbone"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/yolov8/coco/yolov8_x_backbone.h5",  # noqa: E501
        "weights_hash": "ce0cc3235eacaffc4a9824e28b2366e674b6d42befc4c7b77f3be7d1d39960bd",  # noqa: E501
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
