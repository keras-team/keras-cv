# Copyright 2024 The KerasCV Authors
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
import keras
from keras import Sequential
from keras_nlp.layers import RotaryEmbedding
from keras_nlp.layers import TransformerDecoder

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import ops
from keras_cv.layers import TransformerEncoder as CVTransformerEncoder
from keras_cv.models.feature_extractor.coca.coca_layers import CoCaAttentionPooling
from keras_cv.layers.vit_layers import PatchingAndEmbedding
from keras_cv.models.task import Task


@keras_cv_export(["keras_cv.models.coca"])
class CoCa(Task):
    """Contrastive Captioner foundational model implementation.

    This model implements the "Contrastive Captioners are image-Text Foundational Models" by Yu, et al.
    (https://arxiv.org/pdf/2205.01917.pdf). In short, the coca model combines the ideas of Contrastive techniques
    such as CLIP, with Generative Captioning approaches such as SimVLM.

    The architecture of clip can be described as an Image Visual Transformer Encoder in parallel to self-attention-only
    Text Transformer Decoder, the outputs of both of which are passed into a multimodal Transformer Decoder. The
    contrastive loss from the ViT and the uni-modal Text Decoder is combined with a captioning loss from the multi-modal
    Decoder in order to produce the combined total loss.

    Basic Usage:
    ```python

    images = ... # [batch_size, height, width, channel]
    text = ... # [batch_size, text_dim, sequence_length]

    coca = coca()

    # [batch_size, sequence_length, captioning_query_length]
    output = coca(images, text)
    ```

    All default arguments should be consistent with the original paper's details.

    Args:
        img_patch_size: N of each NxN patch generated from linearization of the input images
        encoder_depth: number of image encoder blocks
        encoder_heads: number of attention heads used in each image encoder block
        encoder_intermediate_dim: dimensionality of the encoder blocks' intermediate representation (MLP dimensionality)
        encoder_width: dimensionality of the encoder's projection, consistent with wording used in coca paper.
        unimodal_decoder_depth: number of decoder blocks used for text self-attention/embedding
        multimodal_decoder_depth: number of decoder blocks used for image-text cross-attention and captioning
        decoder_intermediate_dim: dimensionality of the decoder blocks' MLPs
        unimodal_decoder_heads: number of attention heads in the unimodal decoder
        multimodal_decoder_heads: number of attention heads in the multimodal decoder
        contrastive_query_length: number of tokens to use to represent contrastive query
        captioning_query_length: number of tokens to use to represent captioning query
        contrastive_attn_heads: number of attention heads used for the contrastive attention pooling
        captioning_attn_heads: number of attention heads used for the captioning attention pooling
        contrastive_loss_weight: weighting of contrastive loss
        captioning_loss_weight: weighting of captioning loss
    """

    def __init__(
            self,
            img_patch_size=18,
            encoder_depth=40,
            encoder_heads=16,
            encoder_intermediate_dim=6144,
            encoder_width=1408,
            unimodal_decoder_depth=18,
            multimodal_decoder_depth=18,
            decoder_intermediate_dim=5632,
            unimodal_decoder_heads=16,
            multimodal_decoder_heads=16,
            contrastive_query_length=1,
            captioning_query_length=256,
            contrastive_attn_heads=16,
            captioning_attn_heads=16,
            contrastive_loss_weight=0.5,
            captioning_loss_weight=0.5,
            **kwargs,
    ):
        super().__init__(**kwargs)

        #
        # Save Details
        #
        self.img_patch_size = img_patch_size

        self.encoder_depth = encoder_depth
        self.encoder_heads = encoder_heads
        self.encoder_width = encoder_width
        self.encoder_intermediate_dim = encoder_intermediate_dim

        self.unimodal_decoder_depth = unimodal_decoder_depth
        self.multimodal_decoder_depth = multimodal_decoder_depth
        self.decoder_intermediate_dim = decoder_intermediate_dim
        self.unimodal_decoder_heads = unimodal_decoder_heads
        self.multimodal_decoder_heads = multimodal_decoder_heads

        self.contrastive_query_length = contrastive_query_length
        self.contrastive_attn_heads = contrastive_attn_heads
        self.contrastive_loss_weight = contrastive_loss_weight

        self.captioning_query_length = captioning_query_length
        self.captioning_attn_heads = captioning_attn_heads
        self.captioning_loss_weight = captioning_loss_weight

        #
        # Layer Definitions
        #
        self.image_patching = PatchingAndEmbedding(
            self.encoder_width, self.img_patch_size
        )
        self.image_encoder = Sequential(
            [
                CVTransformerEncoder(
                    self.encoder_width,
                    self.encoder_heads,
                    self.encoder_intermediate_dim,
                )
                for _ in range(self.encoder_depth)
            ]
        )

        self.text_embedding = RotaryEmbedding()
        self.unimodal_text_decoder = Sequential(
            [
                TransformerDecoder(
                    self.decoder_intermediate_dim, self.unimodal_decoder_heads
                )
                for _ in range(self.unimodal_decoder_depth)
            ]
        )
        self.multimodal_text_decoders = [
            TransformerDecoder(
                self.decoder_intermediate_dim, self.multimodal_decoder_heads
            )
            for _ in range(self.multimodal_decoder_depth)
        ]

        self.contrastive_attn_pooling = CoCaAttentionPooling(
            self.encoder_width, self.contrastive_attn_heads
        )
        self.captioning_attn_pooling = CoCaAttentionPooling(
            self.encoder_width, self.captioning_attn_heads
        )

        # These are learnable weights defined in build as per Keras recommendations
        self.contrastive_query = None
        self.captioning_query = None

        #
        # Functional Model
        #
        images = keras.Input(
            shape=(None,), dtype="int32", name="images"
        )

        captions = keras.Input(
            shape=(None,), dtype="int32", name="caption"
        )

        img_encoding = self.image_patching(
            images
        )  # [batch_size, img_patches_len+1, encoder_width]
        img_encoding = self.image_encoder(
            img_encoding
        )  # [batch_size, img_patches_len+1, encoder_width]

        # Learnable Weights
        self.contrastive_query = self.add_weight(
            shape=(
                None,
                self.encoder_width,
                self.contrastive_query_length,
            ),
            trainable=True,
        )
        self.captioning_query = self.add_weight(
            shape=(
                None,
                self.encoder_width,
                self.captioning_query_length,
            ),
            trainable=True,
        )

        # This is for contrastive loss; [batch_size, encoder_width, contrastive_query_length]
        contrastive_feature = self.con_attn_pooling(self.contrastive_query, img_encoding)

        # [batch_size, encoder_width, captioning_query_length]
        captioning_feature = self.captioning_attn_pooling(
            self.captioning_query, img_encoding
        )

        # Learnable CLs Token
        self.cls_token = self.add_weight(
            shape=(None, 1, ), name="cls_token", trainable=True
        )

        # [batch_size, sequence_length+1, text_dim]
        text_tokens = ops.concatenate(captions, self.cls_token)
        mask = ops.concatenate(
            (ops.ones_like(captions), ops.zeros_like(self.cls_token))
        )

        # [batch_size, sequence_length+1, text_dim]
        embed_text = self.text_embedding(text_tokens)
        unimodal_out = self.unimodal_text_decoder(
            embed_text, attention_mask=mask
        )

        # [batch_size, sequence_length, captioning_query_length], notice we remove the CLs token
        multimodal_out = unimodal_out[:, :-1, :]
        for decoder in self.multimodal_text_decoders:
            multimodal_out = decoder(
                multimodal_out,
                encoder_sequence=captioning_feature,
                decoder_attention_mask=mask
            )

        super().__init__(
            inputs={
                "images": images,
                "captions": captions,
            },
            outputs={
                "multimodal_out": multimodal_out,
                "contrastive_feature": contrastive_feature
            },
        )


    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "img_patch_size": self.img_patch_size,
                "encoder_depth": self.encoder_depth,
                "encoder_heads": self.encoder_heads,
                "encoder_width": self.encoder_width,
                "encoder_intermediate_dim": self.encoder_intermediate_dim,
                "unimodal_decoder_depth": self.unimodal_decoder_depth,
                "multimodal_decoder_depth": self.multimodal_decoder_depth,
                "decoder_intermediate_dim": self.decoder_intermediate_dim,
                "unimodal_decoder_heads": self.unimodal_decoder_heads,
                "multimodal_decoder_heads": self.multimodal_decoder_heads,
                "contrastive_query_length": self.contrastive_query_length,
                "contrastive_attn_heads": self.contrastive_attn_heads,
                "contrastive_loss_weight": self.contrastive_loss_weight,
                "captioning_query_length": self.captioning_query_length,
                "captioning_attn_heads": self.captioning_attn_heads,
                "captioning_loss_weight": self.captioning_loss_weight,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def load_own_variables(self, store):
        print(store)
        super().load_own_variables(store)