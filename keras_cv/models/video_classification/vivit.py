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

    # Instantiate Model
    model = ViViT(
        projection_dim=PROJECTION_DIM,
        patch_size=PATCH_SIZE,
        inp_shape=INPUT_SHAPE,
        transformer_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        layer_norm_eps=LAYER_NORM_EPS,
        num_classes=NUM_CLASSES,
    )

    # Compile model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )

    # Build Model
    model.build(INPUT_SHAPE)

    # Train Model
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
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.projection_dim = projection_dim
        self.patch_size = patch_size
        self.tubelet_embedder = TubeletEmbedding(
            embed_dim=self.projection_dim, patch_size=self.patch_size
        )

        self.positional_encoder = PositionalEncoder(
            embed_dim=self.projection_dim
        )
        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_eps
        )
        self.attention_output = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim // num_heads,
            dropout=0.1,
        )
        self.dense_1 = keras.layers.Dense(
            units=projection_dim * 4, activation=keras.ops.gelu
        )

        self.dense_2 = keras.layers.Dense(
            units=projection_dim, activation=keras.ops.gelu
        )
        self.add = keras.layers.Add()
        self.pooling = keras.layers.GlobalAvgPool1D()
        self.dense_output = keras.layers.Dense(
            units=num_classes, activation="softmax"
        )

        self.inp_shape = inp_shape
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.projection_dim = projection_dim
        self.patch_size = patch_size
        self.transformer_layers = transformer_layers

    def build(self, input_shape):
        super().build(input_shape)
        self.tubelet_embedder.build(input_shape)
        flattened_patch_shape = self.tubelet_embedder.compute_output_shape(
            input_shape
        )
        self.positional_encoder.build(flattened_patch_shape)
        self.layer_norm.build([None, None, self.projection_dim])
        self.attention_output.build(
            query_shape=[None, None, self.projection_dim],
            value_shape=[None, None, self.projection_dim],
        )
        self.add.build(
            [
                (None, None, self.projection_dim),
                (None, None, self.projection_dim),
            ]
        )

        self.dense_1.build([None, None, self.projection_dim])
        self.dense_2.build([None, None, self.projection_dim * 4])
        self.pooling.build([None, None, self.projection_dim])
        self.dense_output.build([None, self.projection_dim])

    def call(self, x):
        patches = self.tubelet_embedder(x)
        encoded_patches = self.positional_encoder(patches)
        for _ in range(self.transformer_layers):
            x1 = self.layer_norm(encoded_patches)
            attention_output = self.attention_output(x1, x1)
            x2 = self.add([attention_output, encoded_patches])
            x3 = self.layer_norm(x2)
            x4 = self.dense_1(x3)
            x5 = self.dense_2(x4)
            encoded_patches = self.add([x5, x2])
        representation = self.layer_norm(encoded_patches)
        pooled_representation = self.pooling(representation)
        outputs = self.dense_output(pooled_representation)
        return outputs

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
