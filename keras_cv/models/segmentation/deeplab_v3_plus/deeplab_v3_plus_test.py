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

from keras_cv.models import ResNet18V2Backbone, DeepLabV3Plus


class DeepLabV3PlusTest(tf.test.TestCase, parameterized.TestCase):
    @pytest.fixture(autouse=True)
    def cleanup_global_session(self):
        # Code before yield runs before the test
        tf.config.set_soft_device_placement(False)
        yield
        # Reset soft device placement to not interfere with other unit test
        # files
        tf.config.set_soft_device_placement(True)
        keras.backend.clear_session()

    def test_deeplab_v3_plus_construction(self):
        backbone = ResNet18V2Backbone(input_shape=[512, 512, 3])
        model = DeepLabV3Plus(num_classes=1, backbone=backbone)
        model.compile(
            optimizer="adam",
            loss=keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )

    @pytest.mark.large
    def test_deeplab_v3_plus_call(self):
        backbone = ResNet18V2Backbone(input_shape=[512, 512, 3])
        model = DeepLabV3Plus(num_classes=1, backbone=backbone)
        images = tf.random.uniform((2, 512, 512, 3))
        _ = model(images)
        _ = model.predict(images)

    def test_weights_contained_in_trainable_variables(self):
        target_size = [512, 512]
        images = tf.ones(shape=[1] + target_size + [3])

        backbone = ResNet18V2Backbone(input_shape=target_size + [3])
        model = DeepLabV3Plus(num_classes=1, backbone=backbone)

        model.compile(
            optimizer="adam",
            loss=keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )

        variable_names = [x.name for x in model.trainable_variables]
        outputs = model(images)

        # encoder
        self.assertIn("conv2d_1/kernel:0", variable_names)
        # segmentation head
        self.assertIn("segmentation_head_conv/kernel:0", variable_names)
        # Output shape
        self.assertEqual(outputs.shape, tuple([1] + target_size + [1]))

    @pytest.mark.large
    def test_weights_change(self):
        target_size = [512, 512]

        images = tf.ones(shape=[1] + target_size + [3])
        labels = tf.zeros(shape=[1] + target_size + [3])
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        ds = ds.repeat(2)
        ds = ds.batch(2)

        backbone = ResNet18V2Backbone(input_shape=target_size + [3])
        model = DeepLabV3Plus(num_classes=1, backbone=backbone)

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
        target_size = [512, 512]

        backbone = ResNet18V2Backbone(input_shape=target_size + [3])
        model = DeepLabV3Plus(num_classes=1, backbone=backbone)

        input_batch = tf.ones(shape=[2] + target_size + [3])
        model_output = model(input_batch)

        save_path = os.path.join(self.get_temp_dir(), filename)
        model.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, DeepLabV3Plus)

        # Check that output matches.
        restored_output = restored_model(input_batch)
        self.assertAllClose(model_output, restored_output)
