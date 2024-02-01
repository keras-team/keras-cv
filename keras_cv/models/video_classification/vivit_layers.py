# Copyright 2024 The KerasCV Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.backend import ops


@keras_cv_export(
    "keras_cv.layers.TubeletEmebedding",
    package="keras_cv.layers",
)
class TubeletEmbedding(keras.layers.Layer):
    """
    A Keras layer for spatio-temporal tube embedding applied to input sequences
    retrieved from video frames.

    References:
      - [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691)
      (ICCV 2021)

    Args:
    embed_dim: int, number of dimensions in the embedding space.
        Defaults to 128.
    patch_size: tuple or int, size of the spatio-temporal patch.
        If int, the same size is used for all dimensions.
        If tuple, specifies the size for each dimension.
        Defaults to 8.

    """

    def __init__(self, embed_dim=128, patch_size=8, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.patch_size = patch_size

    def build(self, input_shape):
        self.projection = keras.layers.Conv3D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="VALID",
        )
        self.flatten = keras.layers.Reshape(target_shape=(-1, self.embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches

    def get_config(self):
        config = super().get_config()
        config.update(
            {"embed_dim": self.embed_dim, "patch_size": self.patch_size}
        )
        return config


@keras_cv_export(
    "keras_cv.layers.PositionalEncoder",
    package="keras_cv.layers",
)
class PositionalEncoder(keras.layers.Layer):
    """
    A Keras layer for adding positional information to the encoded video tokens.

    References:
      - [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691)
      (ICCV 2021)

    Args:
    embed_dim: int, number of dimensions in the embedding space.
        Defaults to 128.

    """

    def __init__(self, embed_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        _, num_tokens, _ = input_shape
        self.position_embedding = keras.layers.Embedding(
            input_dim=num_tokens, output_dim=self.embed_dim
        )
        self.positions = ops.arange(start=0, stop=num_tokens, step=1)

    def call(self, encoded_tokens):
        encoded_positions = self.position_embedding(self.positions)
        encoded_tokens = encoded_tokens + encoded_positions
        return encoded_tokens
