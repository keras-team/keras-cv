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
"""Tests for ImageClassifier."""


import os

import pytest
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_cv.models.backbones.resnet_v2.resnet_v2_backbone import (
    ResNet18V2Backbone,
)
from keras_cv.models.classification.image_classifier import ImageClassifier


class ImageClassifierTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.input_batch = tf.ones(shape=(8, 224, 224, 3))
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (self.input_batch, tf.one_hot(tf.ones((8,), dtype="int32"), 2))
        ).batch(4)
        self.model = ImageClassifier(
            backbone=ResNet18V2Backbone(),
            num_classes=2,
        )

    def test_valid_call(self):
        model = ImageClassifier(
            backbone=ResNet18V2Backbone(),
            num_classes=2,
        )
        model(self.input_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_classifier_fit(self, jit_compile):
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
            jit_compile=jit_compile,
        )
        self.model.fit(self.dataset)

    @parameterized.named_parameters(
        ("avg_pooling", "avg"), ("max_pooling", "max")
    )
    def test_pooling_arg_call(self, pooling):
        model = ImageClassifier(
            backbone=ResNet18V2Backbone(),
            num_classes=2,
            pooling=pooling,
        )
        model(self.input_batch)

    def test_throw_invalid_pooling(self):
        with self.assertRaises(ValueError):
            ImageClassifier(
                backbone=ResNet18V2Backbone(),
                num_classes=2,
                pooling="clowntown",
            )

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        model = ImageClassifier(
            backbone=ResNet18V2Backbone(),
            num_classes=2,
        )
        model_output = model(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(), filename)
        model.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, ImageClassifier)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)


@pytest.mark.large
class ImageClassifierPresetSmokeTest(tf.test.TestCase, parameterized.TestCase):
    """
    A smoke test for ImageClassifier presets we run continuously.
    This only tests the smallest weights we have available. Run with:
    `pytest keras_cv/models/classification/image_classifier_test.py --run_large`
    """

    def setUp(self):
        self.input_batch = tf.ones(shape=(8, 224, 224, 3))

    @parameterized.named_parameters(
        ("preset_with_weights", "resnet50_v2_imagenet"),
        ("preset_no_weights", "resnet50_v2"),
    )
    def test_backbone_preset_call(self, preset):
        model = ImageClassifier.from_preset(
            preset,
            num_classes=2,
        )
        model(self.input_batch)

    def test_classifier_preset_call(self):
        model = ImageClassifier.from_preset("resnet50_v2_imagenet_classifier")
        model(self.input_batch)


if __name__ == "__main__":
    tf.test.main()
