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
"""ResNetV2 model preset configurations."""

backbone_presets = {
    "resnet18_v2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [2, 2, 2, 2],
        "stackwise_strides": [1, 2, 2, 2],
        "include_rescaling": True,
        "stackwise_dilations": None,
        "input_shape": (None, None, 3),
        "input_tensor": None,
        "pooling": None,
        "block_type": "basic_block",
    },
    "resnet34_v2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [3, 4, 6, 3],
        "stackwise_strides": [1, 2, 2, 2],
        "include_rescaling": True,
        "stackwise_dilations": None,
        "stackwise_dilations": None,
        "input_shape": (None, None, 3),
        "input_tensor": None,
        "pooling": None,
        "block_type": "basic_block",
    },
    "resnet50_v2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [3, 4, 6, 3],
        "stackwise_strides": [1, 2, 2, 2],
        "include_rescaling": True,
        "stackwise_dilations": None,
        "stackwise_dilations": None,
        "input_shape": (None, None, 3),
        "input_tensor": None,
        "pooling": None,
        "block_type": "block",
    },
    "resnet101_v2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [3, 4, 23, 3],
        "stackwise_strides": [1, 2, 2, 2],
        "include_rescaling": True,
        "stackwise_dilations": None,
        "stackwise_dilations": None,
        "input_shape": (None, None, 3),
        "input_tensor": None,
        "pooling": None,
        "block_type": "block",
    },
    "resnet152_v2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [3, 8, 36, 3],
        "stackwise_strides": [1, 2, 2, 2],
        "include_rescaling": True,
        "stackwise_dilations": None,
        "stackwise_dilations": None,
        "input_shape": (None, None, 3),
        "input_tensor": None,
        "pooling": None,
        "block_type": "block",
    },
}
