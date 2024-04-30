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
from tensorflow import data as tf_data

from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.backend.config import keras_3
from keras_cv.src.models import CLIP
from keras_cv.src.models.feature_extractor.clip import CLIPProcessor
from keras_cv.src.tests.test_case import TestCase

VOCAB_PATH = keras.utils.get_file(
    None,
    "https://storage.googleapis.com/keras-cv/models/clip/vocab.json",
)
MERGE_PATH = keras.utils.get_file(
    None,
    "https://storage.googleapis.com/keras-cv/models/clip/merges.txt",
)


@pytest.mark.skipif(
    not keras_3(),
    reason="Only works with Keras 3",
)
class CLIPTest(TestCase):

    @pytest.mark.large
    def test_clip_model_golden_values(self):
        model = CLIP.from_preset("clip-vit-base-patch32")
        processed_image = np.ones(shape=[1, 224, 224, 3])
        processed_text = np.ones(shape=[3, 77])
        attention_mask = np.ones(shape=[3, 77])
        outputs = model(
            {
                "images": processed_image,
                "token_ids": processed_text,
                "padding_mask": attention_mask,
            }
        )

        # These values are NOT computing using HF as the reference model.
        # Currently, the numerics of the CLIP model don't match the
        # HF model exactly (for the same inputs). For the time being,
        # these tests just confirm that unrelated changed don't affect
        # the numerics. Once the fix for the numerics is in, we can remove
        # this comment and the xfail below.
        self.assertAllClose(
            outputs["image_logits"], [[10.246354, 10.246353, 10.246354]]
        )
        self.assertAllClose(
            outputs["text_logits"],
            ops.transpose([[10.246354, 10.246353, 10.246354]]),
        )

        # True reference values computed using HF:
        # image_logits: [[17.8013, 17.8013, 17.8013]]
        # text_logits: image_logits.T

        # xfail after assertion
        pytest.xfail("KerasCV CLIP doesn't match the HF model.")

    def test_clip_preprocessor(self):
        processor = CLIPProcessor(VOCAB_PATH, MERGE_PATH)
        tokens = processor(["mountains", "cat on tortoise"])
        self.assertAllClose(
            tokens["token_ids"][:, :3],
            [[49406, 5873, 49407], [49406, 2368, 525]],
        )
        self.assertAllClose(
            tokens["padding_mask"][0, :5], [True, True, True, False, False]
        )

    def test_clip_preprocessor_tf_data(self):
        processor = CLIPProcessor(VOCAB_PATH, MERGE_PATH)
        text_input = ["a bus", "a dog", "a cat"]
        dataset = tf_data.Dataset.from_tensor_slices(text_input)
        dataset.map(processor)

    @pytest.mark.large
    def test_presets(self):
        model = CLIP.from_preset("clip-vit-base-patch32")
        processed_image = np.ones(shape=[1, 224, 224, 3])
        processed_text = np.ones(shape=[3, 77])
        attention_mask = np.ones(shape=[3, 77])
        model(
            {
                "images": processed_image,
                "token_ids": processed_text,
                "padding_mask": attention_mask,
            }
        )

    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self):
        model = CLIP.from_preset("clip-vit-base-patch32")
        processed_image = np.ones(shape=[1, 224, 224, 3])
        processed_text = np.ones(shape=[3, 77])
        attention_mask = np.ones(shape=[3, 77])
        outputs = model(
            {
                "images": processed_image,
                "token_ids": processed_text,
                "padding_mask": attention_mask,
            }
        )
        save_path = os.path.join(self.get_temp_dir(), "model.keras")
        if keras_3():
            model.save(save_path)
        else:
            model.save(save_path, save_format="keras_v3")
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, CLIP)
        # Check that output matches.
        restored_outputs = restored_model(
            {
                "images": processed_image,
                "token_ids": processed_text,
                "padding_mask": attention_mask,
            }
        )
        self.assertAllClose(outputs, restored_outputs)
