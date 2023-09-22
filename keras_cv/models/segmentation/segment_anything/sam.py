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

import copy

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.models.backbones.backbone_presets import backbone_presets
from keras_cv.models.backbones.backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.models.segmentation.segment_anything.sam_presets import (
    sam_presets,
)
from keras_cv.models.task import Task
from keras_cv.utils.python_utils import classproperty


@keras_cv_export(
    "keras_cv.models.SegmentAnythingModel", package="keras_cv.models"
)
class SegmentAnythingModel(Task):
    """
    The Segment Anything (SAM) Model.

    Args:
        backbone (keras_cv.models.Backbone): A feature extractor for the input
            images.
        prompt_encoder (keras_cv.models.SAMPromptEncoder): A Keras layer to
            compute embeddings for points, box, and mask prompt.
        mask_decoder (keras_cv.models.SAMMaskDecoder): A Keras layer to
            generate segmentation masks given the embeddings generated by the
            backbone and the prompt encoder.

    References:
        - [Segment Anything paper](https://arxiv.org/abs/2304.02643)
        - [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything)

    Examples:

    >>> import numpy as np
    >>> from keras_cv.models import ViTDetBBackbone
    >>> from keras_cv.models import SAMPromptEncoder
    >>> from keras_cv.models import SAMMaskDecoder

    Create all the components of the SAM model:

    >>> backbone = ViTDetBBackbone()
    >>> prompt_encoder = SAMPromptEncoder()
    >>> mask_decoder = SAMMaskDecoder()

    Instantiate the model:

    >>> sam = SegmentAnythingModel(
    ...     backbone=backbone,
    ...     prompt_encoder=prompt_encoder,
    ...     mask_decoder=mask_decoder
    ... )

    Define the input of the backbone. This must be a batch of images of shape
    `(1024, 1024, 3)` for the ViT backbone we are using:

    >>> image = np.ones((1, 1024, 1024, 3))

    SAM works by prompting the input images. There are three ways to prompt:

    (1) Labelled Points: Foreground points (points with label 1) are encoded
        such that the output masks generated by the mask decoder contain them
        and background points (points with label 0) are encoded such that the
        generated masks don't contain them.
    (2) Box: A box tells the model which part/crop of the image to segment.
    (3) Mask: An input mask can be used to refine the output of the mask
        decoder.

    These prompts can be mixed and matched but at least one of the prompts
    must be present. To turn off a particular prompt, simply exclude it from
    the inputs to the model.

    # TODO(ianstenbit): Remove the need for the `1` axes, and fix the box shape.
    (1) For points prompts, the expected shape is `(batch, num_points, 2)`.
        The labels must have a corresponding shape of `(batch, num_points)`.
    (2) For box prompt, the expected shape is `(batch, 1, 2, 2)`.
    (3) Similarly, mask prompts have shape `(batch, 1, H, W, 1)`.

    For example, to pass in all the prompts, do:

    >>> points = np.array([[[512., 512.], [100., 100.]]])
    >>> # For labels: 1 means foreground point, 0 means background
    >>> labels = np.array([[1., 0.]])
    >>> box = np.array([[[[384., 384.], [640., 640.]]]])
    >>> input_mask = np.ones((1, 1, 256, 256, 1))

    Prepare an input dictionary:

    >>> inputs = {
    ...     "images": image,
    ...     "points": points,
    ...     "labels": labels,
    ...     "boxes": box,
    ...     "masks": input_mask
    ... }
    ...
    >>> outputs = sam.predict(inputs)
    >>> masks, iou_pred = outputs["masks"], outputs["iou_pred"]

    The first mask in the output `masks` (i.e. `masks[:, 0, ...]`) is the best
    mask predicted by the model based on the prompts. Other `masks`
    (i.e. `masks[:, 1:, ...]`) are alternate predictions that can be used if
    they are desired over the first one.

    Now, in case of only points and box prompts, simply exclude the masks:

    >>> inputs = {
    ...     "images": image,
    ...     "points": points,
    ...     "labels": labels,
    ...     "boxes": box,
    ...     "masks": no_input_mask
    ... }
    ...
    >>> outputs = sam.predict(inputs)
    >>> masks, iou_pred = outputs["masks"], outputs["iou_pred"]

    # TODO(ianstenbit): Remove the need for this padding.
    Another example is that only points prompts are present.
    Note that if point prompts are present but no box prompt is present, the
    points must be padded using a zero point and -1 label:

    >>> padded_points = np.concatenate(
    ...     [points, np.zeros((1, 1, 2))], axis=1
    ... )
    ...
    >>> padded_labels = np.concatenate(
    ...     [labels, -np.ones((1, 1))], axis=1
    ... )
    >>> inputs = {
    ...     "images": image,
    ...     "points": padded_points,
    ...     "labels": padded_labels,
    ... }
    ...
    >>> outputs = sam.predict(inputs)
    >>> masks, iou_pred = outputs["masks"], outputs["iou_pred"]

    Note that the segment anything model only supports inference and training
    isn't support yet. So, calling the `fit` method will fail for now.
    """  # noqa: E501

    def __init__(self, *, backbone, prompt_encoder, mask_decoder, **kwargs):
        # Get the image encoder input -- Images
        backbone_input = backbone.input

        # Define the prompt encoder inputs -- Prompts
        prompt_inputs = {
            "points": keras.Input(shape=[None, 2], name="points"),
            "labels": keras.Input(shape=[None], name="labels"),
            "boxes": keras.Input(shape=[None, 2, 2], name="boxes"),
            "masks": keras.Input(shape=[None, None, None, 1], name="masks"),
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

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy({**backbone_presets, **sam_presets})

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return copy.deepcopy({**backbone_presets_with_weights, **sam_presets})

    @classproperty
    def backbone_presets(cls):
        """Dictionary of preset names and configurations of compatible
        backbones."""
        return copy.deepcopy(backbone_presets)
