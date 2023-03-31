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

"""ConvMixer model preset configurations."""

backbone_presets_no_weights = {
    "convmixer_1536_20": {
        "metadata": {
            "description": (
                "ConvMixer model with 20 layers where the layer normalization "
                "and GELU activation are applied after the convolution layers "
                "with 1536 output channels."
            ),
        },
        "class_name": "keras_cv.models>ConvMixerBackbone",
        "config": {
            "dim": 1536,
            "depth": 20,
            "patch_size": 7,
            "kernel_size": 9,
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "convmixer_1536_24": {
        "metadata": {
            "description": (
                "ConvMixer model with 24 layers where the layer normalization "
                "and GELU activation are applied after the convolution layers "
                "with 1536 output channels."
            ),
        },
        "class_name": "keras_cv.models>ConvMixerBackbone",
        "config": {
            "dim": 1536,
            "depth": 24,
            "patch_size": 14,
            "kernel_size": 9,
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "convmixer_768_32": {
        "metadata": {
            "description": (
                "ConvMixer model with 32 layers where the layer normalization "
                "and GELU activation are applied after the convolution layers "
                "with 768 output channels."
            ),
        },
        "class_name": "keras_cv.models>ConvMixerBackbone",
        "config": {
            "dim": 768,
            "depth": 32,
            "patch_size": 7,
            "kernel_size": 7,
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "convmixer_1024_16": {
        "metadata": {
            "description": (
                "ConvMixer model with 16 layers where the layer normalization "
                "and GELU activation are applied after the convolution layers "
                "with 1024 output channels."
            ),
        },
        "class_name": "keras_cv.models>ConvMixerBackbone",
        "config": {
            "dim": 1024,
            "depth": 16,
            "patch_size": 7,
            "kernel_size": 9,
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
    "convmixer_512_16": {
        "metadata": {
            "description": (
                "ConvMixer model with 16 layers where the layer normalization "
                "and GELU activation are applied after the convolution layers "
                "with 512 output channels."
            ),
        },
        "class_name": "keras_cv.models>ConvMixerBackbone",
        "config": {
            "dim": 512,
            "depth": 16,
            "patch_size": 7,
            "kernel_size": 8,
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
        },
    },
}

backbone_presets_with_weights = {
    "convmixer_512_16": {
        "metadata": {
            "description": (
                "ConvMixer model with 16 layers where the layer normalization "
                "and GELU activation are applied after the convolution layers "
                "with 512 output channels."
                "Trained on Imagenet 2012 classification task."
            ),
        },
        "class_name": "keras_cv.models>ConvMixerBackbone",
        "config": backbone_presets_no_weights["convmixer_512_16"]["config"],
        "weights_url": "https://storage.googleapis.com/keras-cv/models/convmixer_512_16/imagenet/classification-v0-notop.h5",
        "weights_hash": "aa08c7fa9ca6ec045c4783e1248198dbe1bc141e2ae788e712de471c0370822c",
    },
}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}
