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
"""DarkNet model preset configurations."""

backbone_presets_no_weights = {
    "darknet21": {
        "metadata": {
            "description": "DarkNet model with 21 layers.",
        },
        "class_name": "keras_cv.models>DarkNetBackbone",
        "config": {
            "stackwise_blocks": [1, 2, 2, 1],
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "darknet53": {
        "metadata": {
            "description": "DarkNet model with 53 layers.",
        },
        "class_name": "keras_cv.models>DarkNetBackbone",
        "config": {
            "stackwise_blocks": [2, 8, 8, 4],
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
}

backbone_presets_with_weights = {
    "darknet53_imagenet": {
        "metadata": {
            "description": (
                "DarkNet model with 53 layers. "
                "Trained on Imagenet 2012 classification task."
            ),
        },
        "class_name": "keras_cv.models>DarkNetBackbone",
        "config": backbone_presets_no_weights["darknet53"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/darknet53/imagenet/classification-v0-notop.h5",  # noqa: E501
        "weights_hash": "8dcce43163e4b4a63e74330ba1902e520211db72d895b0b090b6bfe103e7a8a5",  # noqa: E501
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
