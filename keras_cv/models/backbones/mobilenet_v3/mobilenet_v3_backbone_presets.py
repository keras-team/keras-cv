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

"""MobileNetV3 model preset configurations."""

backbone_presets_no_weights = {
    "mobilenetv3small": {
        "metadata": {
            "description": (
                "MobileNetV3 model with 14 layers where the batch "
                "normalization and hard-swish activation are applied after the "
                "convolution layers."
            ),
        },
        "class_name": "keras_cv.models>MobileNetV3Backbone",
        "config": {
            "last_point_ch": 1024,
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "alpha": 1.0,
            "minimalistic": True,
            "dropout_rate": 0.2,
        },
    },
    "mobilenetv3large": {
        "metadata": {
            "description": (
                "MobileNetV3 model with 28 layers where the batch "
                "normalization and hard-swish activation are applied after the "
                "convolution layers."
            ),
        },
        "class_name": "keras_cv.models>MobileNetV3Backbone",
        "config": {
            "last_point_ch": 1280,
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "alpha": 1.0,
            "minimalistic": True,
            "dropout_rate": 0.2,
        },
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
}
