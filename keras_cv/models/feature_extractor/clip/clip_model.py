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
from keras_cv.backend import ops
from keras_cv.models.feature_extractor.clip.clip_image_model import (
    CLIPImageEncoder,
)
from keras_cv.models.feature_extractor.clip.clip_presets import (  # noqa: E501
    clip_presets,
)
from keras_cv.models.feature_extractor.clip.clip_text_model import (
    CLIPTextEncoder,
)
from keras_cv.models.task import Task
from keras_cv.utils.conditional_imports import assert_keras_nlp_installed
from keras_cv.utils.python_utils import classproperty

try:
    import keras_nlp
except ImportError:
    keras_nlp = None


@keras_cv_export(["keras_cv.models.CLIP"])
class CLIP(Task):
    """
    CLIP implements the Contrastive Language-Image Pretraining (CLIP)
    architecture, which enables joint learning of visual and textual
    representations for various downstream tasks. The deafult base model
    achitecture will be set to clip-vit-base-patch32.

    Args:
        embed_dim (int): The dimensionality of the joint embedding space for
            images and texts.
        image_resolution (int): The resolution of the input images (both height
            and width).
        vision_layers (int): The number of layers in the vision (image) encoder.
            vision_width (int): The width of the hidden layers in the vision
            encoder.
        vision_patch_size (int): The size of each square patch in the input
            images.
        context_length (int): The maximum length of the contextualized text
            sequences.
        vocab_size (int): The size of the vocabulary for tokenization.
        transformer_width (int): The width of the hidden layers in the
            transformer-based text encoder.
        transformer_heads (int): The number of attention heads in the
            transformer-based text encoder.
        transformer_layers (int): The number of layers in the transformer-based
            text encoder.
    Example:
    ```python
    processor = CLIPProcessor(
      input_resolution=224,
      "path_to_vocab.json",
      "path_to_merges.txt"
      )
    processed_image = processor.process_images(["cat.jpg"])
    processed_text, attention_mask = processor.process_texts(
      ["mountains", "cat on tortoise", "two cats"]
      )
    model = CLIP.from_preset("clip-vit-base-patch16")
    image_logits, text_logits = model(
            {
                "image": processed_image,
                "text": processed_text,
                "attention_mask": attention_mask,
            }
        )
    ```
    """

    def __init__(
        self,
        embed_dim=512,
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert_keras_nlp_installed("CLIP")
        self.embed_dim = embed_dim
        self.image_resolution = image_resolution
        self.vision_layers = vision_layers
        self.vision_width = vision_width
        self.vision_patch_size = vision_patch_size
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers

        vision_heads = self.vision_width // 64
        self.image_encoder = CLIPImageEncoder(
            input_resolution=self.image_resolution,
            patch_size=self.vision_patch_size,
            width=self.vision_width,
            num_layers=self.vision_layers,
            heads=vision_heads,
            output_dim=self.embed_dim,
            name="image_encoder",
        )
        self.text_encoder = CLIPTextEncoder(
            transformer_width=self.transformer_width,
            transformer_layers=self.transformer_layers,
            transformer_heads=self.transformer_heads,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            context_length=self.context_length,
            name="text_encoder",
        )

        self.logit_scale = keras.Variable(
            ops.ones([]) * ops.log(1 / 0.07), name="logit_scale"
        )
        self.image_embeddings = None
        self.text_embeddings = None

    def build(self, input_shape):
        super().build(input_shape)
        self.text_encoder.build([None, self.context_length])
        self.image_encoder.build(
            [None, self.image_resolution, self.image_resolution, 3]
        )

    def encode_images(self, image):
        return self.image_encoder(image)

    def encode_text(self, text, attention_mask=None):
        return self.text_encoder(text, attention_mask=attention_mask)

    def call(self, inputs):
        image, text = inputs["image"], inputs["text"]
        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"]
        else:
            attention_mask = None
        self.image_embeddings = self.encode_images(image)
        self.text_embeddings = self.encode_text(
            text, attention_mask=attention_mask
        )
        normalize_image_features = ops.sqrt(
            ops.sum(ops.power(self.image_embeddings, 2), keepdims=True)
        )
        normalize_text_features = ops.sqrt(
            ops.sum(ops.power(self.text_embeddings, 2), keepdims=True)
        )
        self.image_embeddings = self.image_embeddings / normalize_image_features
        self.text_embeddings = self.text_embeddings / normalize_text_features
        logit_scale = ops.exp(self.logit_scale)
        logits_per_image = (
            ops.matmul(
                self.image_embeddings,
                ops.transpose(self.text_embeddings),
            )
            * logit_scale
        )
        logits_per_text = ops.transpose(logits_per_image)

        return logits_per_image, logits_per_text

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy({**clip_presets})

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return copy.deepcopy({**clip_presets})

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "image_resolution": self.image_resolution,
                "vision_layers": self.vision_layers,
                "vision_width": self.vision_width,
                "vision_patch_size": self.vision_patch_size,
                "context_length": self.context_length,
                "vocab_size": self.vocab_size,
                "transformer_width": self.transformer_width,
                "transformer_heads": self.transformer_heads,
                "transformer_layers": self.transformer_layers,
            }
        )
        return config
