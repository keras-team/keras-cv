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
import numpy as np
from keras import Sequential
from keras_cv.api_export import keras_cv_export
from keras_nlp.layers import RotaryEmbedding, TransformerDecoder
from keras_cv.layers import TransformerEncoder as CVTransformerEncoder
from keras_cv.models.task import Task
from keras_cv.layers.attention_pooling import AttentionPooling
from keras_cv.layers.vit_layers import PatchingAndEmbedding


@keras_cv_export(["keras_cv.models.CoCa"])
class CoCa(Task):
    def __init__(self,
                 img_query_dim,
                 text_proj_dim,
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
                 con_queries=1,
                 cap_queries=256,
                 con_heads=16,
                 cap_heads=16,
                 cap_loss_weight=0.5,
                 con_loss_weight=0.5,
                 **kwargs):
        super().__init__(**kwargs)

        self.img_patch_size = img_patch_size
        self.img_query_dim = img_query_dim

        self.encoder_depth = encoder_depth
        self.encoder_heads = encoder_heads
        self.encoder_width = encoder_width
        self.encoder_intermediate_dim = encoder_intermediate_dim

        self.text_proj_dim = text_proj_dim
        self.unimodal_decoder_depth = unimodal_decoder_depth
        self.multimodal_decoder_depth = multimodal_decoder_depth
        self.decoder_intermediate_dim = decoder_intermediate_dim
        self.unimodal_decoder_heads = unimodal_decoder_heads
        self.multimodal_decoder_heads = multimodal_decoder_heads

        self.con_queries = con_queries
        self.con_heads = con_heads
        self.con_loss_weight = con_loss_weight

        self.cap_queries = cap_queries
        self.cap_heads = cap_heads
        self.cap_loss_weight = cap_loss_weight

    def build(self, input_shape):
        super().build(input_shape)

        self.image_patching = PatchingAndEmbedding(self.encoder_width, self.img_patch_size)
        self.image_encoder = Sequential([
            CVTransformerEncoder(self.img_query_dim, self.encoder_heads, self.encoder_intermediate_dim)
            for _ in range(self.encoder_depth)
        ])

        self.cls_token = self.add_weight(shape=[1, 1, self.text_proj_dim], name="cls_token", trainable=True)

        self.text_embedding = RotaryEmbedding()
        self.unimodal_text_decoder = Sequential([
            TransformerDecoder(self.decoder_intermediate_dim, self.unimodal_decoder_heads)
            for _ in range(self.unimodal_decoder_depth)
        ])
        self.multimodal_text_decoder = Sequential([
            TransformerDecoder(self.decoder_intermediate_dim, self.multimodal_decoder_heads)
            for _ in range(self.multimodal_decoder_depth)
        ])

        self.con_query = self.add_weight(shape=[1, 1, self.con_queries], trainable=True)
        self.cap_query = self.add_weight(shape=[1, 1, self.cap_queries], trainable=True)

        self.con_attn_pooling = AttentionPooling(self.img_query_dim, self.con_heads)
        self.cap_attn_pooling = AttentionPooling(self.img_query_dim, self.cap_heads)

    def call(self, images, texts):
        """
        Forward pass of the Coca Model

        :param images: [batch_size, height, width, channels] representing images
        :param texts: Tensor, typically represented as [batch_size, sequence_length, feature_length] or
            [batch_size, sequence_length, num_heads, feature_length]. The sequence_length and/or feature_length
            are required.
        :return: output of the captioning Transformer Decoder with captioning cross-attention
        """
        img_encoding = self.image_patching(images)
        img_encoding = self.image_encoder(img_encoding)  # [batch, patches_len+1, img_query_dim]

        # This is only needed for loss calculations
        # con_feature = self.con_attn_pooling(self.con_query, img_encoding)
        cap_feature = self.cap_attn_pooling(self.cap_query, img_encoding)

        text_tokens = np.concatenate(texts, self.cls_token)
        mask = np.concatenate((np.ones_like(texts), np.zeros_like(self.cls_token)))

        embed_text = self.text_embedding(text_tokens)
        unimodal_out = self.unimodal_text_decoder(embed_text, attention_mask=mask)
        multimodal_out = self.multimodal_text_decoder(unimodal_out[:, :-1, :],
                                                      encoder_sequence=cap_feature,
                                                      decoder_attention_mask=mask)

        return multimodal_out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "img_query_dim": self.img_query_dim,
                "img_patch_size": self.img_patch_size,
                "text_proj_dim": self.text_proj_dim,
                "cap_loss_weight": self.cap_loss_weight,
                "con_loss_weight": self.con_loss_weight
            }
        )
        return config
