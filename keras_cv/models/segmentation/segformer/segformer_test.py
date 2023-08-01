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

import pytest
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_cv.models import MiTBackbone
from keras_cv.models import SegFormer


class SegFormerTest(tf.test.TestCase, parameterized.TestCase):
    def test_segformer_construction(self):
        backbone = MiTBackbone.from_preset("mit_b0", input_shape=[512, 512, 3])
        model = SegFormer(backbone=backbone, num_classes=1)
        model.compile(
            optimizer="adam",
            loss=keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )

    @pytest.mark.large
    def test_segformer_plus_call(self):
        backbone = MiTBackbone.from_preset("mit_b0", input_shape=[512, 512, 3])
        model = SegFormer(backbone=backbone, num_classes=1)
        images = tf.random.uniform((2, 512, 512, 3))
        _ = model(images)
        _ = model.predict(images)

    @pytest.mark.large
    def test_weights_change(self):
        target_size = [512, 512, 3]

        images = tf.ones(shape=[1] + target_size)
        labels = tf.zeros(shape=[1] + target_size)
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        ds = ds.repeat(2)
        ds = ds.batch(2)

        backbone = MiTBackbone.from_preset("mit_b0", input_shape=[512, 512, 3])
        model = SegFormer(backbone=backbone, num_classes=1)

        model.compile(
            optimizer="adam",
            loss=keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )

        original_weights = model.get_weights()
        model.fit(ds, epochs=1)
        updated_weights = model.get_weights()

        for w1, w2 in zip(original_weights, updated_weights):
            self.assertNotAllClose(w1, w2)
            self.assertFalse(tf.math.reduce_any(tf.math.is_nan(w2)))

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self, save_format, filename):
        target_size = [512, 512, 3]

        backbone = MiTBackbone.from_preset("mit_b0", input_shape=[512, 512, 3])
        model = SegFormer(backbone=backbone, num_classes=1)

        input_batch = tf.ones(shape=[2] + target_size)
        model_output = model(input_batch)

        save_path = os.path.join(self.get_temp_dir(), filename)
        model.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, SegFormer)

        # Check that output matches.
        restored_output = restored_model(input_batch)
        self.assertAllClose(model_output, restored_output)
