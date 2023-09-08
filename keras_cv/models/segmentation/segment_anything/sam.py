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

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.models.backbones.detectron2.detectron2_backbone import (
    ViTDetBackbone,
)
from keras_cv.models.segmentation.segment_anything.sam_mask_decoder import (
    SAMMaskDecoder,
)
from keras_cv.models.segmentation.segment_anything.sam_prompt_encoder import (
    SAMPromptEncoder,
)
from keras_cv.models.task import Task


@keras_cv_export("keras_cv.models.SegmentAnythingModel")
class SegmentAnythingModel(Task):
    def __init__(
        self,
        *,
        backbone: ViTDetBackbone,
        prompt_encoder: SAMPromptEncoder,
        mask_decoder: SAMMaskDecoder,
        **kwargs
    ):
        if not isinstance(backbone, ViTDetBackbone):
            raise ValueError("`backbone` must be a `ViTDetBackbone`.")
        if not isinstance(prompt_encoder, SAMPromptEncoder):
            raise ValueError("`prompt_encoder` must be a `SAMPromptEncoder`.")
        if not isinstance(mask_decoder, SAMMaskDecoder):
            raise ValueError("`mask_decoder` must be a `SAMMaskDecoder`.")

        # Get the image encoder input -- Images
        backbone_input = backbone.input

        # Define the prompt encoder inputs -- Prompts
        prompt_inputs = {
            "points": keras.Input(shape=[None, 2], name="points"),
            "labels": keras.Input(shape=[None], name="labels"),
            "box": keras.Input(shape=[None, 2, 2], name="box"),
            "mask": keras.Input(shape=[None, None, None, 1], name="mask"),
        }

        # All Inputs -- Images + Prompts
        all_inputs = {"images": backbone_input}
        all_inputs.update(prompt_inputs)

        # Build the prompt encoder
        prompt_embeddings = prompt_encoder(prompt_inputs)

        # Define the mask decoder inputs
        mask_decoder_inputs = {
            "image_embeddings": backbone.output,
            "image_pe": prompt_embeddings["dense_positional_embeddings"],
            "sparse_prompt_embeddings": prompt_embeddings["sparse_embeddings"],
            "dense_prompt_embeddings": prompt_embeddings["dense_embeddings"],
        }

        # Build the mask decoder
        outputs = mask_decoder(mask_decoder_inputs)

        super().__init__(inputs=all_inputs, outputs=outputs, **kwargs)

        self.backbone = backbone
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            "Segment Anything Model only supports inference for now. Training"
            " the model isn't supported yet."
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone": keras.saving.serialize_keras_object(self.backbone),
                "prompt_encoder": keras.saving.serialize_keras_object(
                    self.prompt_encoder
                ),
                "mask_decoder": keras.saving.serialize_keras_object(
                    self.mask_decoder
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config.update(
            {
                "prompt_encoder": keras.layers.deserialize(
                    config["prompt_encoder"]
                ),
                "mask_decoder": keras.layers.deserialize(
                    config["mask_decoder"]
                ),
            }
        )
        return super().from_config(config)
