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
"""Tests for VideoClassifier."""


import os

import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.models.backbones.video_swin.video_swin_backbone import (
    VideoSwinBackbone,  # TODO: update with aliases (kaggle handle)
)
from keras_cv.models.classification.video_classifier import VideoClassifier
from keras_cv.tests.test_case import TestCase


class VideoClassifierTest(TestCase):
    def setUp(self):
        self.input_batch = np.ones(shape=(10, 8, 224, 224, 3))
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (self.input_batch, tf.one_hot(tf.ones((10,), dtype="int32"), 10))
        ).batch(4)

    def test_valid_call(self):
        model = VideoClassifier(
            backbone=VideoSwinBackbone(
                input_shape=(8, 224, 224, 3), include_rescaling=False
            ),
            num_classes=10,
        )
        model(self.input_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    @pytest.mark.large  # Fit is slow, so mark these large.
    def test_classifier_fit(self, jit_compile):
        model = VideoClassifier(
            backbone=VideoSwinBackbone(
                input_shape=(8, 224, 224, 3), include_rescaling=True
            ),
            num_classes=10,
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
            jit_compile=jit_compile,
        )
        model.fit(self.dataset)

    @parameterized.named_parameters(
        ("avg_pooling", "avg"), ("max_pooling", "max")
    )
    def test_pooling_arg_call(self, pooling):
        model = VideoClassifier(
            backbone=VideoSwinBackbone(
                input_shape=(8, 224, 224, 3), include_rescaling=True
            ),
            num_classes=10,
            pooling=pooling,
        )
        model(self.input_batch)

    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self):
        model = VideoClassifier(
            backbone=VideoSwinBackbone(
                input_shape=(8, 224, 224, 3), include_rescaling=False
            ),
            num_classes=10,
        )
        model_output = model(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(), "video_classifier.keras")
        model.save(save_path)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, VideoClassifier)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(
            ops.convert_to_numpy(model_output),
            ops.convert_to_numpy(restored_output),
        )


if __name__ == "__main__":
    tf.test.main()
