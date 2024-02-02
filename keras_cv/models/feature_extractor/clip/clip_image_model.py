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

from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.models.feature_extractor.clip.clip_encoder import CLIPEncoder
from keras_cv.models.feature_extractor.clip.clip_encoder import get_initializer


class CLIPPatchingAndEmbedding(keras.layers.Layer):
    def __init__(
        self, width, patch_size, input_resolution, output_dim, **kwargs
    ):
        super().__init__(**kwargs)

        self.conv1 = keras.layers.Conv2D(
            filters=width,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            use_bias=False,
            data_format="channels_last",
            kernel_initializer=get_initializer(0.02),
            name="patch_embed.embedding",
        )
        self.width = width
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.num_patches = ops.power(
            (self.input_resolution // self.patch_size), 2
        )
        self.class_embedding_initializer = get_initializer(
            ops.power(self.width, -0.5) * 0.02
        )
        self.output_dim = output_dim

    def build(self, input_shape):
        self.conv1.build(input_shape)
        self.class_embedding = self.add_weight(
            shape=((self.width,)),
            initializer=self.class_embedding_initializer,
            name="patch_embed.class_embedding",
        )

        self.positional_embedding = self.add_weight(
            shape=(
                (
                    (self.input_resolution // self.patch_size) ** 2 + 1,
                    self.width,
                )
            ),
            trainable=True,
            name="patch_embed.positional_embedding",
        )

    def call(self, x):
        batch_size, _, _, _ = ops.shape(x)
        patch_embeddings = self.conv1(x)  # shape = [*, grid, grid, channel]

        patch_embeddings = ops.reshape(
            patch_embeddings, (batch_size, self.num_patches, -1)
        )
        class_embeds = ops.broadcast_to(
            self.class_embedding, (batch_size, 1, self.width)
        )
        embeddings = ops.concatenate(
            [class_embeds, patch_embeddings], axis=1
        )  # shape = [*, grid ** 2 + 1, width]
        positional_embedding = self.positional_embedding
        embeddings = embeddings + positional_embedding
        return embeddings


class CLIPImageEncoder(keras.Model):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.input_resolution = input_resolution
        self.width = width
        self.patch_size = patch_size
        self.output_dim = output_dim

        self.embeddings = CLIPPatchingAndEmbedding(
            width=self.width,
            patch_size=self.patch_size,
            input_resolution=self.input_resolution,
            output_dim=self.output_dim,
        )
        self.pre_norm = keras.layers.LayerNormalization(
            epsilon=1e-5, name="ln_1"
        )
        self.encoder = CLIPEncoder(
            width,
            layers,
            heads,
            name="clip_encoder",
        )
        self.post_norm = keras.layers.LayerNormalization(
            epsilon=1e-5, name="ln_2"
        )
        self.image_projector = keras.layers.Dense(
            output_dim, name="vision_projector", use_bias=False
        )

    def build(self, input_shape):
        self.embeddings.build(input_shape)

    def call(self, image):
        embeddings = self.embeddings(image)
        pre_norm = self.pre_norm(embeddings)
        encoded_output = self.encoder(pre_norm)
        post_norm = self.post_norm(encoded_output[:, 0, :])
        image_projected_embeddings = self.image_projector(post_norm)
        return image_projected_embeddings
