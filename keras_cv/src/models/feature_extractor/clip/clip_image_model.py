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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.models.feature_extractor.clip.clip_encoder import CLIPEncoder


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
            name="patch_embed.embedding",
        )
        self.width = width
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.num_patches = ops.power(
            (self.input_resolution // self.patch_size), 2
        )
        self.output_dim = output_dim

    def build(self, input_shape):
        super().build(input_shape)
        self.conv1.build(input_shape)
        self.class_embedding = self.add_weight(
            shape=((self.width,)),
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

    def compute_output_shape(self, input_shape):
        return [
            None,
            (self.input_resolution // self.patch_size) ** 2 + 1,
            self.width,
        ]

    def call(self, x):
        batch_size = ops.shape(x)[0]
        patch_embeddings = self.conv1(x)  # shape = [*, grid, grid, channel]

        patch_embeddings = ops.reshape(
            patch_embeddings, (batch_size, self.num_patches, -1)
        )
        class_embeds = ops.broadcast_to(
            self.class_embedding.value, (batch_size, 1, self.width)
        )
        embeddings = ops.concatenate(
            [class_embeds, patch_embeddings], axis=1
        )  # shape = [*, grid ** 2 + 1, width]
        positional_embedding = self.positional_embedding
        embeddings = embeddings + positional_embedding
        return embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "width": self.width,
                "patch_size": self.patch_size,
                "input_resolution": self.input_resolution,
                "output_dim": self.output_dim,
            }
        )
        return config


@keras_cv_export("keras_cv.models.feature_extractor.CLIPImageEncoder")
class CLIPImageEncoder(keras.Model):
    def __init__(
        self,
        input_resolution,
        patch_size,
        width,
        num_layers,
        heads,
        output_dim,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.input_resolution = input_resolution
        self.width = width
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.heads = heads
        self.num_layers = num_layers

        self.embeddings = CLIPPatchingAndEmbedding(
            width=self.width,
            patch_size=self.patch_size,
            input_resolution=self.input_resolution,
            output_dim=self.output_dim,
            name="clip_patch_embedding",
        )
        self.pre_norm = keras.layers.LayerNormalization(
            epsilon=1e-5, name="ln_1"
        )
        self.encoder = CLIPEncoder(
            self.width,
            self.num_layers,
            self.heads,
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
        self.pre_norm.build([None, None, self.width])
        self.encoder.build(None)
        self.post_norm.build([None, self.width])
        self.image_projector.build([None, self.width])
        self.built = True

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.output_dim]

    def call(self, image):
        x = self.embeddings(image)
        x = self.pre_norm(x)
        x = self.encoder(x)
        x = self.post_norm(x[:, 0, :])
        image_projected_embeddings = self.image_projector(x)
        return image_projected_embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_resolution": self.input_resolution,
                "patch_size": self.patch_size,
                "width": self.width,
                "layers": self.num_layers,
                "heads": self.heads,
                "output_dim": self.output_dim,
            }
        )
        return config
