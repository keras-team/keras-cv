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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.models.feature_extractor.clip.clip_image_model import (
    CLIPImageEncoder,
)
from keras_cv.src.models.feature_extractor.clip.clip_presets import (  # noqa: E501
    clip_presets,
)
from keras_cv.src.models.feature_extractor.clip.clip_text_model import (
    CLIPTextEncoder,
)
from keras_cv.src.models.task import Task
from keras_cv.src.utils.python_utils import classproperty

try:
    import keras_nlp
except ImportError:
    keras_nlp = None


class CLIPHead(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.logit_scale = self.add_variable(
            shape=(),
            initializer=lambda *a, **kw: ops.log(1 / 0.07),
            trainable=True,
            dtype=self.variable_dtype,
            name="logit_scale",
        )
        self.built = True

    def call(self, image_embeddings, text_embeddings):
        normalize_image_features = ops.sqrt(
            ops.sum(ops.power(image_embeddings, 2), keepdims=True)
        )
        normalize_text_features = ops.sqrt(
            ops.sum(ops.power(text_embeddings, 2), keepdims=True)
        )
        image_embeddings = image_embeddings / normalize_image_features
        text_embeddings = text_embeddings / normalize_text_features
        logit_scale = ops.exp(self.logit_scale)
        image_logits = (
            ops.matmul(
                image_embeddings,
                ops.transpose(text_embeddings),
            )
            * logit_scale
        )
        text_logits = ops.transpose(image_logits)
        return image_logits, text_logits


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
    tokens = processor(
        ["mountains", "cat on tortoise", "two cats"]
    )
    model = CLIP.from_preset("clip-vit-base-patch16")
    image_logits, text_logits = model(
        {
            "images": processed_image,
            "token_ids": tokens["token_ids"],
            "padding_mask": tokens["padding_mask"],
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
        if keras_nlp is None:
            raise ValueError(
                "ClipTokenizer requires keras-nlp. Please install "
                "using pip `pip install -U keras-nlp && pip install -U keras`"
            )

        vision_heads = vision_width // 64

        images = keras.Input(
            shape=[image_resolution, image_resolution, 3], name="images"
        )
        token_ids = keras.Input(
            shape=[
                context_length,
            ],
            name="token_ids",
        )
        padding_mask = keras.Input(
            shape=[
                context_length,
            ],
            name="padding_mask",
        )

        image_encoder = CLIPImageEncoder(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            num_layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            name="image_encoder",
        )
        text_encoder = CLIPTextEncoder(
            transformer_width=transformer_width,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            context_length=context_length,
            name="text_encoder",
        )
        clip_head = CLIPHead(name="clip_head")

        image_embeddings = image_encoder(images)
        text_embeddings = text_encoder(token_ids, attention_mask=padding_mask)
        image_logits, text_logits = clip_head(image_embeddings, text_embeddings)

        inputs = {
            "images": images,
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
        outputs = {
            "image_logits": image_logits,
            "text_logits": text_logits,
        }

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

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
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.clip_head = clip_head

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
