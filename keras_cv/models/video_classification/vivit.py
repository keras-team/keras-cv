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
from keras_cv.models.video_classification.vivit_layers import PositionalEncoder
from keras_cv.models.video_classification.vivit_layers import TubeletEmbedding


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
        inp_shape: tuple, the shape of the input video frames.
        num_classes: int, the number of classes for video classification.
        transformer_layers: int, the number of transformer layers in the model.
            Defaults to 8.
        patch_size: tuple , contains the size of the
        spatio-temporal patches for each dimension
            Defaults to (8,8,8)
        num_heads: int, the number of heads for multi-head
            self-attention mechanism. Defaults to 8.
        projection_dim: int, number of dimensions in the projection space.
            Defaults to 128.
        layer_norm_eps: float, epsilon value for layer normalization.
            Defaults to 1e-6.


    Examples:
    ```python
    import keras_cv

    INPUT_SHAPE = (32, 32, 32, 1)
    NUM_CLASSES = 11
    PATCH_SIZE = (8, 8, 8)
    LAYER_NORM_EPS = 1e-6
    PROJECTION_DIM = 128
    NUM_HEADS = 8
    NUM_LAYERS = 8

    frames = np.random.uniform(size=(5, 32, 32, 32, 1))
    labels = np.ones(shape=(5))

    model = ViViT(
        projection_dim=PROJECTION_DIM,
        patch_size=PATCH_SIZE,
        inp_shape=INPUT_SHAPE,
        transformer_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        embed_dim=PROJECTION_DIM,
        layer_norm_eps=LAYER_NORM_EPS,
        num_classes=NUM_CLASSES,
    )

    # Evaluate model
    model(frames)

    # Train model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )

    model.fit(frames, labels, epochs=3)

    ```
    """

    def __init__(
        self,
        inp_shape,
        num_classes,
        projection_dim=128,
        patch_size=(8, 8, 8),
        transformer_layers=8,
        num_heads=8,
        embed_dim=128,
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        self.projection_dim = projection_dim
        self.patch_size = patch_size
        self.tubelet_embedder = TubeletEmbedding(
            embed_dim=self.projection_dim, patch_size=self.patch_size
        )

        self.positional_encoder = PositionalEncoder(
            embed_dim=self.projection_dim
        )

        inputs = keras.layers.Input(shape=inp_shape)
        patches = self.tubelet_embedder(inputs)
        encoded_patches = self.positional_encoder(patches)

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

        self.inp_shape = inp_shape
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.projection_dim = projection_dim
        self.patch_size = patch_size

    def build(self, input_shape):
        self.tubelet_embedder.build(input_shape)
        flattened_patch_shape = self.tubelet_embedder.compute_output_shape(
            input_shape
        )
        self.positional_encoder.build(flattened_patch_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "inp_shape": self.inp_shape,
                "num_classes": self.num_classes,
                "projection_dim": self.projection_dim,
                "patch_size": self.patch_size,
            }
        )
        return config
