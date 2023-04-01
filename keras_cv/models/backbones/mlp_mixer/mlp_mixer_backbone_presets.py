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

"""MLPMixer model preset configurations."""

backbone_presets_no_weights = {
    "mlpmixerb16": {
        "metadata": {
            "description": (
                "MLPMixer model with 16x16 patches of resolution and with 24 "
                "layers where the layer normalization and GELU activation are "
                "applied after the convolution layers."
            ),
        },
        "class_name": "keras_cv.models>MLPMixerBackbone",
        "config": {
            "patch_size": 16,
            "num_blocks": 12,
            "hidden_dim": 768,
            "tokens_mlp_dim": 384,
            "channels_mlp_dim": 3072,
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "mlpmixerb32": {
        "metadata": {
            "description": (
                "MLPMixer model with 32x32 patches of resolution and with 24 "
                "layers where the layer normalization and GELU activation "
                "are applied after the convolution layers."
            ),
        },
        "class_name": "keras_cv.models>MLPMixerBackbone",
        "config": {
            "patch_size": 32,
            "num_blocks": 12,
            "hidden_dim": 768,
            "tokens_mlp_dim": 384,
            "channels_mlp_dim": 3072,
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "mlpmixerl16": {
        "metadata": {
            "description": (
                "MLPMixer model with 16x16 patches of resolution and with 48 "
                "layers where the layer normalization and GELU activation are "
                "applied after the convolution layers."
            ),
        },
        "class_name": "keras_cv.models>MLPMixerBackbone",
        "config": {
            "patch_size": 16,
            "num_blocks": 24,
            "hidden_dim": 1024,
            "tokens_mlp_dim": 512,
            "channels_mlp_dim": 4096,
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
}
