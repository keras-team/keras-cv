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
"""SAM model preset configurations."""

from keras_cv.models.backbones.vit_det import vit_det_backbone_presets

prompt_encoder_preset = {
    "class_name": "keras_cv.models>SAMPromptEncoder",
    "config": {
        "embed_dim": 256,
        "image_embedding_size": (64, 64),
        "input_image_size": (1024, 1024),
        "mask_in_chans": 16,
        "activation": "gelu",
    },
}

mask_decoder_preset = {
    "class_name": "keras_cv.models>SAMMaskDecoder",
    "config": {
        "transformer_dim": 256,
        "transformer": {
            "class_name": "keras_cv.models>TwoWayTransformer",
            "config": {
                "depth": 2,
                "embed_dim": 256,
                "num_heads": 8,
                "mlp_dim": 2048,
                "activation": "relu",
                "attention_downsample_rate": 2,
            },
        },
        "num_multimask_outputs": 3,
        "iou_head_depth": 3,
        "iou_head_hidden_dim": 256,
        "activation": "gelu",
    },
}

sam_presets = {
    "sam_base_sa1b": {
        "metadata": {
            "description": "The base SAM model trained on the SA1B dataset.",
            "params": 93_735_728,
            "official_name": "SAM",
            "path": "segment_anything",
        },
        "config": {
            "backbone": vit_det_backbone_presets.backbone_presets[
                "vitdet_base"
            ],
            "prompt_encoder": prompt_encoder_preset,
            "mask_decoder": mask_decoder_preset,
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/segment_anything/sam_base.h5",  # noqa: E501
        "weights_hash": "5a18868ed227b6f093d4a6cb7ed689868dd11f288a8311ae69002a9a9d86d192",  # noqa: E501
    },
    "sam_large_sa1b": {
        "metadata": {
            "description": "The large SAM model trained on the SA1B dataset.",
            "params": 312_343_088,
            "official_name": "SAM",
            "path": "segment_anything",
        },
        "config": {
            "backbone": vit_det_backbone_presets.backbone_presets[
                "vitdet_large"
            ],
            "prompt_encoder": prompt_encoder_preset,
            "mask_decoder": mask_decoder_preset,
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/segment_anything/sam_large.h5",  # noqa: E501
        "weights_hash": "4ef43d3a8e24200c14a086a043dec8e956fef500c6171268a35029ea720305f0",  # noqa: E501
    },
    "sam_huge_sa1b": {
        "metadata": {
            "description": "The huge SAM model trained on the SA1B dataset.",
            "params": 641_090_864,
            "official_name": "SAM",
            "path": "segment_anything",
        },
        "config": {
            "backbone": vit_det_backbone_presets.backbone_presets[
                "vitdet_huge"
            ],
            "prompt_encoder": prompt_encoder_preset,
            "mask_decoder": mask_decoder_preset,
        },
        "weights_url": "https://storage.googleapis.com/keras-cv/models/segment_anything/sam_huge.h5",  # noqa: E501
        "weights_hash": "3284c7c3c91274e8cb1ec2de69da3b6d6cee4483f7d8b0e17e1042b9dfc86fe5",  # noqa: E501
    },
}
