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
"""ConvNeXT model preset configurations."""

backbone_presets_no_weights = {
    "convnext_tiny": {
        "metadata": {
            "description": "DESCRIPTION",
        },
        "class_name": "keras_cv.models>ConvNeXtBackbone",
        "config": {
            "depths": [3, 3, 9, 3],
            "projection_dims": [96, 192, 384, 768],
            "default_size": 224,
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "convnext_small": {
        "metadata": {
            "description": "DESCRIPTION",
        },
        "class_name": "keras_cv.models>ConvNeXtBackbone",
        "config": {
            "depths": [3, 3, 27, 3],
            "projection_dims": [96, 192, 384, 768],
            "default_size": 224,
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "convnext_base": {
        "metadata": {
            "description": "DESCRIPTION",
        },
        "class_name": "keras_cv.models>ConvNeXtBackbone",
        "config": {
            "depths": [3, 3, 27, 3],
            "projection_dims": [128, 256, 512, 1024],
            "default_size": 224,
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "convnext_large": {
        "metadata": {
            "description": "DESCRIPTION",
        },
        "class_name": "keras_cv.models>ConvNeXtBackbone",
        "config": {
            "depths": [3, 3, 27, 3],
            "projection_dims": [192, 384, 768, 1536],
            "default_size": 224,
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "convnext_xlarge": {
        "metadata": {
            "description": "DESCRIPTION",
        },
        "class_name": "keras_cv.models>ConvNeXtBackbone",
        "config": {
            "depths": [3, 3, 27, 3],
            "projection_dims": [256, 512, 1024, 2048],
            "default_size": 224,
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
}

backbone_presets_with_weights = {}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
