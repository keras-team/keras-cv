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

import os

import numpy as np
import pytest
import tensorflow as tf

from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.backend.config import keras_3
from keras_cv.models.video_classification.vivit import ViViT
from keras_cv.tests.test_case import TestCase


class ViViT_Test(TestCase):
    def test_vivit_construction(self):
        input_shape = (28, 28, 28, 1)
        num_classes = 11
        patch_size = (8, 8, 8)
        layer_norm_eps = 1e-6
        projection_dim = 128
        num_heads = 8
        num_layers = 8

        model = ViViT(
            projection_dim=projection_dim,
            patch_size=patch_size,
            inp_shape=input_shape,
            transformer_layers=num_layers,
            num_heads=num_heads,
            layer_norm_eps=layer_norm_eps,
            num_classes=num_classes,
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
        input_shape = (28, 28, 28, 1)
        num_classes = 11
        patch_size = (8, 8, 8)
        layer_norm_eps = 1e-6
        projection_dim = 128
        num_heads = 8
        num_layers = 8

        model = ViViT(
            projection_dim=projection_dim,
            patch_size=patch_size,
            inp_shape=input_shape,
            transformer_layers=num_layers,
            num_heads=num_heads,
            layer_norm_eps=layer_norm_eps,
            num_classes=num_classes,
        )
        model.build(input_shape)
        frames = np.random.uniform(size=(5, 28, 28, 28, 1))
        _ = model(frames)

    def test_weights_change(self):
        input_shape = (28, 28, 28, 1)
        num_classes = 11
        patch_size = (8, 8, 8)
        layer_norm_eps = 1e-6
        projection_dim = 128
        num_heads = 8
        num_layers = 8

        frames = np.random.uniform(size=(5, 28, 28, 28, 1))
        labels = np.ones(shape=(5))
        ds = tf.data.Dataset.from_tensor_slices((frames, labels))
        ds = ds.repeat(2)
        ds = ds.batch(2)

        model = ViViT(
            projection_dim=projection_dim,
            patch_size=patch_size,
            inp_shape=input_shape,
            transformer_layers=num_layers,
            num_heads=num_heads,
            layer_norm_eps=layer_norm_eps,
            num_classes=num_classes,
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
        model.build(input_shape)
        representation_layer = model.get_layer(index=-8)  # Accesses MHSA Layer
        original_weights = representation_layer.get_weights()
        model.fit(ds, epochs=1)
        updated_weights = representation_layer.get_weights()

        for w1, w2 in zip(original_weights, updated_weights):
            self.assertNotAllEqual(w1, w2)
            self.assertFalse(ops.any(ops.isnan(w2)))

    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self):
        input_shape = (28, 28, 28, 1)
        num_classes = 11
        patch_size = (8, 8, 8)
        layer_norm_eps = 1e-6
        projection_dim = 128
        num_heads = 8
        num_layers = 8

        model = ViViT(
            projection_dim=projection_dim,
            patch_size=patch_size,
            inp_shape=input_shape,
            transformer_layers=num_layers,
            num_heads=num_heads,
            layer_norm_eps=layer_norm_eps,
            num_classes=num_classes,
        )
        model.build(input_shape)
        input_batch = np.random.uniform(size=(5, 28, 28, 28, 1))
        model_output = model(input_batch)

        save_path = os.path.join(self.get_temp_dir(), "model.keras")
        if keras_3():
            model.save(save_path)
        else:
            model.save(save_path, save_format="keras_v3")
        restored_model = keras.models.load_model(save_path)

        self.assertIsInstance(restored_model, ViViT)

        restored_output = restored_model(input_batch)
        self.assertAllClose(model_output, restored_output)
