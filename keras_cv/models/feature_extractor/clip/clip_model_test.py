# Copyright 2022 The KerasCV Authors
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

from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.backend.config import keras_3
from keras_cv.models import CLIP
from keras_cv.models.feature_extractor.clip import CLIPProcessor
from keras_cv.tests.test_case import TestCase

VOCAB_PATH = keras.utils.get_file(
    None,
    "https://storage.googleapis.com/keras-cv/models/clip/vocab.json",
)
MERGE_PATH = keras.utils.get_file(
    None,
    "https://storage.googleapis.com/keras-cv/models/clip/merges.txt",
)

MODEL_PATH = keras.utils.get_file(
    None,
    "https://storage.googleapis.com/keras-cv/models/clip/clip-vit-base-patch32.weights.h5",  # noqa: E501
)


@pytest.mark.tf_only
class CLIPTest(TestCase):
    @pytest.mark.large
    def test_clip_model_golden_values(self):
        model = CLIP()
        model.load_weights(MODEL_PATH)
        processed_image = np.ones(shape=[1, 224, 224, 3])
        processed_text = np.ones(shape=[3, 77])
        attention_mask = np.ones(shape=[3, 77])
        image_logits, text_logits = model(
            processed_image, processed_text, attention_mask
        )
        print(image_logits)
        self.assertAllClose(image_logits, [[2.932678, 2.932678, 2.932675]])
        self.assertAllClose(
            text_logits, ops.transpose([[2.932678, 2.932678, 2.932675]])
        )

    def test_clip_preprocessor(self):
        self.skipTest("TODO: Enable after Kaggle model is public")
        processor = CLIPProcessor(224, VOCAB_PATH, MERGE_PATH)
        processed_text, attention_mask = processor.process_texts(
            ["mountains", "cat on tortoise", "two cats"]
        )

    @pytest.mark.large
    def test_presets(self):
        self.skipTest("TODO: Enable after Kaggle model is public")
        model = CLIP.from_preset("clip-vit-base-patch32")
        processed_image = np.ones(shape=[1, 224, 224, 3])
        processed_text = np.ones(shape=[3, 77])
        attention_mask = np.ones(shape=[3, 77])
        image_logits, text_logits = model(
            processed_image, processed_text, attention_mask
        )

    @pytest.mark.large
    def test_image_encoder_golden_values(self):
        model = CLIP()
        model.load_weights(MODEL_PATH)
        processed_image = np.ones(shape=[1, 224, 224, 3])
        processed_text = np.ones(shape=[3, 77])
        attention_mask = np.ones(shape=[3, 77])
        model(processed_image, processed_text, attention_mask)
        self.assertAllClose(
            model.image_embeddings[:, :5],
            [[0.023215, 0.026526, 0.008914, -0.091689, 0.021791]],
        )

    @pytest.mark.large
    def test_text_encoder_golden_values(self):
        model = CLIP()
        processed_image = np.ones(shape=[1, 224, 224, 3])
        processed_text = np.ones(shape=[3, 77])
        attention_mask = np.ones(shape=[3, 77])
        model(processed_image, processed_text, attention_mask)
        print(model.text_embeddings)
        self.assertAllClose(
            model.text_embeddings[0, :3],
            [-0.018502, 0.000906, 0.020372],
        )

    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self):
        model = CLIP()
        processed_image = np.ones(shape=[1, 224, 224, 3])
        processed_text = np.ones(shape=[3, 77])
        attention_mask = np.ones(shape=[3, 77])
        model_output, _ = model(processed_image, processed_text, attention_mask)
        save_path = os.path.join(self.get_temp_dir(), "model.keras")
        if keras_3():
            model.save(save_path)
        else:
            model.save(save_path, save_format="keras_v3")
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, CLIP)

        # Check that output matches.
        restored_output, _ = restored_model(
            processed_image, processed_text, attention_mask
        )
        self.assertAllClose(model_output, restored_output)
