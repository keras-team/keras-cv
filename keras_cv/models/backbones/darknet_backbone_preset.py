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
"""darknetV1 model preset configurations."""

backbone_presets_no_weights = {
    "darknet121": {
        "metadata": {
            "description": (
                "darknet model with 18 layers where the batch normalization "
                "and ReLU activation are applied after the convolution layers "
                "(v1 style)."
            ),
        },
    }
        "class_name": "keras_cv.models>DarknetBackbone",
        "config": {
            "stackwise_filters": [64, 128, 256, 512],
            "stackwise_blocks": [2, 2, 2, 2],
            "stackwise_strides": [1, 2, 2, 2],
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "block_type": "basic_block",
        },
    },
    "darknet34": {
        "metadata": {
            "description": (
                "darknet model with 34 layers where the batch normalization "
                "and ReLU activation are applied after the convolution layers "
                "(v1 style)."
            ),
        },
        "class_name": "keras_cv.models>DarknetBackbone",
        "config": {
            "stackwise_filters": [64, 128, 256, 512],
            "stackwise_blocks": [3, 4, 6, 3],
            "stackwise_strides": [1, 2, 2, 2],
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "block_type": "basic_block",
        },
    },
    
    

backbone_presets_with_weights = {
    "darknet121_imagenet": {
        "metadata": {
            "description": (
                "darknet model with 50 layers where the batch normalization "
                "and ReLU activation are applied after the convolution layers "
                "(v1 style). "
                "Trained on Imagenet 2012 classification task."
            ),
        },
        "class_name": "keras_cv.models>darknetBackbone",
        "config": backbone_presets_no_weights["darknet121"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/darknet50/imagenet/classification-v0-notop.h5",
        "weights_hash": "dc5f6d8f929c78d0fc192afecc67b11ac2166e9d8b9ef945742368ae254c07af",
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}