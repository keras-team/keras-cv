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

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.models.task import Task


@keras_cv_export(
    [
        "keras_cv.models.ViViT",
        "keras_cv.models.video_classification.ViViT",
    ]
)
class ViViT(Task):
    """A Keras model implementing a Video Vision Transformer
    for video classification.
    References:
      - [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691)
      (ICCV 2021)

    Args:
    #Example
    tubelet_embedder =
        keras_cv.layers.TubeletEmbedding(
              embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
          )
    positional_encoder =
        keras_cv.layers.PositionalEncoder(
              embed_dim=PROJECTION_DIM
          )
    model = keras_cv.models.video_classification.ViViT(
          tubelet_embedder,
          positional_encoder
      )

    """

    def __init__(
        self,
        tubelet_embedder,
        positional_encoder,
        input_shape,
        transformer_layers,
        num_heads,
        embed_dim,
        layer_norm_eps,
        num_classes,
        **kwargs,
    ):
        if not isinstance(tubelet_embedder, keras.layers.Layer):
            raise ValueError(
                "Argument `tubelet_embedder` must be a "
                " `keras.layers.Layer` instance "
                f" . Received instead "
                f"tubelet_embedder={tubelet_embedder} "
                f"(of type {type(tubelet_embedder)})."
            )

        if not isinstance(positional_encoder, keras.layers.Layer):
            raise ValueError(
                "Argument `positional_encoder` must be a "
                "`keras.layers.Layer` instance "
                f" . Received instead "
                f"positional_encoder={positional_encoder} "
                f"(of type {type(positional_encoder)})."
            )

        inputs = keras.layers.Input(shape=input_shape)
        patches = tubelet_embedder(inputs)
        encoded_patches = positional_encoder(patches)

        for _ in range(transformer_layers):
            x1 = keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embed_dim // num_heads,
                dropout=0.1,
            )(x1, x1)

            x2 = keras.layers.Add()([attention_output, encoded_patches])

            x3 = keras.layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = keras.Sequential(
                [
                    keras.layers.Dense(
                        units=embed_dim * 4, activation=keras.ops.gelu
                    ),
                    keras.layers.Dense(
                        units=embed_dim, activation=keras.ops.gelu
                    ),
                ]
            )(x3)

            encoded_patches = keras.layers.Add()([x3, x2])

        representation = keras.layers.LayerNormalization(
            epsilon=layer_norm_eps
        )(encoded_patches)
        representation = keras.layers.GlobalAvgPool1D()(representation)

        outputs = keras.layers.Dense(units=num_classes, activation="softmax")(
            representation
        )

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        self.num_heads = num_heads
        self.num_classes = num_classes
        self.tubelet_embedder = tubelet_embedder
        self.positional_encoder = positional_encoder

    def get_config(self):
        return {
            "num_heads": self.num_heads,
            "num_classes": self.num_classes,
            "tubelet_embedder": keras.saving.serialize_keras_object(
                self.tubelet_embedder
            ),
            "positional_encoder": keras.saving.serialize_keras_object(
                self.positional_encoder
            ),
        }

    @classmethod
    def from_config(cls, config):
        if "tubelet_embedder" in config and isinstance(
            config["tubelet_embedder"], dict
        ):
            config["tubelet_embedder"] = keras.layers.deserialize(
                config["tubelet_embedder"]
            )
        if "positional_encoder" in config and isinstance(
            config["positional_encoder"], dict
        ):
            config["positional_encoder"] = keras.layers.deserialize(
                config["positional_encoder"]
            )
        return super().from_config(config)
