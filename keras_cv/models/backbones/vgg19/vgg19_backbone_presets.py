`# Copyright 2023 The KerasCV Authors
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

"""VGG19 model preset configurations."""

backbone_presets_no_weights = {
    "vgg19": {
        "metadata": {
            "description": (
                "VGG19 model with 19 layers where the ReLU activation is "
                "applied after the convolution layers."
            ),
        },
        "class_name": "keras_cv.models>VGG19Backbone",
        "config": {
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
}
