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

import os

import numpy as np
import pytest
import tensorflow as tf

from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.backend.config import keras_3
from keras_cv.models import BASNet
from keras_cv.models import ResNet34Backbone

from keras_cv.tests.test_case import TestCase


class BASNetTest(TestCase):

    def test_basnet_construction(self):
        backbone = ResNet34Backbone()
        model = BASNet(
            input_shape=[288, 288, 3],
            backbone=backbone,
            num_classes=1
        )
        model.compile(
            optimizer="adam",
            loss=keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )

    @pytest.mark.large
    def test_basnet_call(self):
        backbone = ResNet34Backbone()
        model = BASNet(
            input_shape=[288, 288, 3],
            backbone=backbone,
            num_classes=1
        )
        images = np.random.uniform(size=(2, 288, 288, 3))
        _ = model(images)
        _ = model.predict(images)

    @pytest.mark.large
    def test_weights_change(self):
        input_size = [288, 288, 3]
        target_size = [288, 288, 1]

        images = np.ones([1] + input_size)
        labels = np.random.uniform(size=[1] + target_size)
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        ds = ds.repeat(2)
        ds = ds.batch(2)

        backbone = ResNet34Backbone()
        model = BASNet(
            input_shape=[288, 288, 3],
            backbone=backbone,
            num_classes=1
        )

        model.compile(
            optimizer="adam",
            loss=keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],

        )

        original_weights = model.refinement_head.get_weights()
        model.fit(ds, epochs=1)
        updated_weights = model.refinement_head.get_weights()

        for w1, w2 in zip(original_weights, updated_weights):
            self.assertNotAllEqual(w1, w2)
            self.assertFalse(ops.any(ops.isnan(w2)))

    @pytest.mark.large
    def test_saved_model(self):
        target_size = [288, 288, 3]

        backbone = ResNet34Backbone()
        model = BASNet(
            input_shape=[288, 288, 3],
            backbone=backbone,
            num_classes=1
        )

        input_batch = np.ones(shape=[2] + target_size)
        model_output = model(input_batch)

        save_path = os.path.join(self.get_temp_dir(), "model.keras")
        if keras_3():
            model.save(save_path)
        else:
            model.save(save_path, save_format="keras_v3")
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, BASNet)

        # Check that output matches.
        restored_output = restored_model(input_batch)
        self.assertAllClose(model_output, restored_output)
