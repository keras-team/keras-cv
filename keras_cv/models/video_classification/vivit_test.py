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
import pytest

from keras_cv.backend import keras
from keras_cv.models.video_classification.vivit import ViViT
from keras_cv.models.video_classification.vivit_layers import PositionalEncoder
from keras_cv.models.video_classification.vivit_layers import TubeletEmbedding
from keras_cv.tests.test_case import TestCase


class ViViT_Test(TestCase):
    def test_vivit_construction(self):
        INPUT_SHAPE = (28, 28, 28, 1)
        NUM_CLASSES = 11
        PATCH_SIZE = (8, 8, 8)
        LAYER_NORM_EPS = 1e-6
        PROJECTION_DIM = 128
        NUM_HEADS = 8
        NUM_LAYERS = 8

        model = ViViT(
            tubelet_embedder=TubeletEmbedding(
                embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
            ),
            positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
            input_shape=INPUT_SHAPE,
            transformer_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            embed_dim=PROJECTION_DIM,
            layer_norm_eps=LAYER_NORM_EPS,
            num_classes=NUM_CLASSES,
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(
                    5, name="top-5-accuracy"
                ),
            ],
        )

    def test_vivit_call(self):
        INPUT_SHAPE = (28, 28, 28, 1)
        NUM_CLASSES = 11
        PATCH_SIZE = (8, 8, 8)
        LAYER_NORM_EPS = 1e-6
        PROJECTION_DIM = 128
        NUM_HEADS = 8
        NUM_LAYERS = 8

        model = ViViT(
            tubelet_embedder=TubeletEmbedding(
                embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
            ),
            positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
            input_shape=INPUT_SHAPE,
            transformer_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            embed_dim=PROJECTION_DIM,
            layer_norm_eps=LAYER_NORM_EPS,
            num_classes=NUM_CLASSES,
        )
        frames = np.random.uniform(size=(5, 28, 28, 28, 1))
        _ = model(frames)
