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
from keras_cv.models.feature_extractors.clip.clip_modelling import (
    CLIPPatchingAndEmbedding,
)
from keras_cv.models.feature_extractors.clip.clip_modelling import (
    ResidualTransformerEncoder,
)


class CLIPImageEncoder(keras.Model):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        input_tensor=None,
        **kwargs,
    ):
        inputs = keras.layers.Input(
            tensor=input_tensor, shape=(input_resolution, input_resolution, 3)
        )
        x = inputs

        x = CLIPPatchingAndEmbedding(
            width=width,
            patch_size=patch_size,
            input_resolution=input_resolution,
        )(x)
        x = keras.layers.LayerNormalization(epsilon=1e-6, name="ln_1")(x)

        x = ops.transpose(x, axes=(1, 0, 2))
        x = ResidualTransformerEncoder(
            width,
            layers,
            heads,
            name="residual_transformer_encoder",
        )(x)
        x = ops.transpose(x, axes=(1, 0, 2))

        x = keras.layers.LayerNormalization(epsilon=1e-6, name="ln_2")(
            x[:, 0, :]
        )

        proj = keras.layers.Dense(output_dim, name="vision_projector")
        x = proj(x)

        output = x

        super().__init__(
            inputs=inputs,
            outputs=output,
            **kwargs,
        )
