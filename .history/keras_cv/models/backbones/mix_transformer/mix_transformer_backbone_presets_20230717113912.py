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
    "mobilenet_v3_small": {
        "metadata": {
            "description": (
                "MobileNetV3 model with 14 layers where the batch "
                "normalization and hard-swish activation are applied after the "
                "convolution layers."
            ),
            "params": 933502,
            "official_name": "MobileNetV3",
            "path": "mobilenetv3",
        },
        "class_name": "keras_cv.models>MobileNetV3Backbone",
        "config": {
            "stackwise_expansion": [
                1,
                72.0 / 16,
                88.0 / 24,
                4,
                6,
                6,
                3,
                3,
                6,
                6,
                6,
            ],
            "stackwise_filters": [16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96],
            "stackwise_kernel_size": [3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5],
            "stackwise_stride": [2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1],
            "stackwise_se_ratio": [
                0.25,
                None,
                None,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
                0.25,
            ],
            "stackwise_activation": [
                "relu",
                "relu",
                "relu",
                "hard_swish",
                "hard_swish",
                "hard_swish",
                "hard_swish",
                "hard_swish",
                "hard_swish",
                "hard_swish",
                "hard_swish",
            ],
            "include_rescaling": True,
            "input_shape": (None, None, 3),
            "input_tensor": None,
            "alpha": 1.0,
        },
    },
    "mobilenet_v3_large": {
        "metadata": {
            "description": (
                "MobileNetV3 model with 28 layers where the batch "
                "normalization and hard-swish activation are applied after the "
                "convolution layers."
            ),
            "params": 2994518,
            "official_name": "MobileNetV3",
            "path": "mobilenetv3",
        },
        "class_name": "keras_cv.models>MobileNetV3Backbone",
        "config": {
            "embedding_dims": [32, 64, 160, 256],
            "depths": [2, 2, 2, 2],
        },
    },
}

backbone_presets_with_weights = {}

backbone_presets = {
    **backbone_presets_no_weights,
    **backbone_presets_with_weights,
}


MODEL_CONFIGS = {
    "B0": {"embedding_dims": [32, 64, 160, 256], "depths": [2, 2, 2, 2]},
    "B1": {"embedding_dims": [64, 128, 320, 512], "depths": [2, 2, 2, 2]},
    "B2": {"embedding_dims": [64, 128, 320, 512], "depths": [3, 4, 6, 3]},
    "B3": {"embedding_dims": [64, 128, 320, 512], "depths": [3, 4, 18, 3]},
    "B4": {"embedding_dims": [64, 128, 320, 512], "depths": [3, 8, 27, 3]},
    "B5": {"embedding_dims": [64, 128, 320, 512], "depths": [3, 6, 40, 3]},
}

MODEL_BACKBONES = {"tensorflow": __MiTTF, "pytorch": __MiTPT}


def MiTB0(
    backend,
    include_top,
    classes=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    as_backbone=False,
    **kwargs,
):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )
    return model_class(
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        embed_dims=MODEL_CONFIGS["B0"]["embedding_dims"],
        depths=MODEL_CONFIGS["B0"]["depths"],
        classes=classes,
        include_top=include_top,
        as_backbone=as_backbone,
        **kwargs,
    )


def MiTB1(
    backend,
    include_top,
    classes=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    as_backbone=False,
    **kwargs,
):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )
    return model_class(
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        embed_dims=MODEL_CONFIGS["B1"]["embedding_dims"],
        depths=MODEL_CONFIGS["B1"]["depths"],
        classes=classes,
        include_top=include_top,
        as_backbone=as_backbone,
        **kwargs,
    )


def MiTB2(
    backend,
    include_top,
    classes=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    as_backbone=False,
    **kwargs,
):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )
    return model_class(
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        embed_dims=MODEL_CONFIGS["B2"]["embedding_dims"],
        depths=MODEL_CONFIGS["B2"]["depths"],
        classes=classes,
        include_top=include_top,
        as_backbone=as_backbone,
        **kwargs,
    )


def MiTB3(
    backend,
    include_top,
    classes=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    as_backbone=False,
    **kwargs,
):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )
    return model_class(
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        embed_dims=MODEL_CONFIGS["B3"]["embedding_dims"],
        depths=MODEL_CONFIGS["B3"]["depths"],
        classes=classes,
        include_top=include_top,
        as_backbone=as_backbone,
        **kwargs,
    )


def MiTB4(
    backend,
    include_top,
    classes=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    as_backbone=False,
    **kwargs,
):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )
    return model_class(
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        embed_dims=MODEL_CONFIGS["B4"]["embedding_dims"],
        depths=MODEL_CONFIGS["B4"]["depths"],
        classes=classes,
        include_top=include_top,
        as_backbone=as_backbone,
        **kwargs,
    )


def MiTB5(
    backend,
    include_top,
    classes=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    as_backbone=False,
    **kwargs,
):
    model_class = MODEL_BACKBONES.get(backend)
    if model_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {MODEL_BACKBONES.keys()}"
        )
    return model_class(
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        embed_dims=MODEL_CONFIGS["B5"]["embedding_dims"],
        depths=MODEL_CONFIGS["B5"]["depths"],
        classes=classes,
        include_top=include_top,
        as_backbone=as_backbone,
        **kwargs,
    )
