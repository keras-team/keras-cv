# Copyright 2023 The KerasNLP Authors
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

import json
import os
import numpy as np
import pytest
from absl.testing import parameterized
from keras_cv.models import ImageClassifier
from keras_cv.tests.test_case import TestCase
from keras_cv.utils import preset_utils


class PresetUtilsTest(TestCase):
    @parameterized.parameters(
        (ImageClassifier, "resnet50_v2_imagenet_classifier"),
        (ImageClassifier, "efficientnetv2_s_imagenet_classifier"),
        (ImageClassifier, "mobilenet_v3_large_imagenet_classifier"),
    )
    @pytest.mark.large
    def test_classifier_preset_saving(self, cls, preset_name):
        save_dir = self.get_temp_dir()
        model = cls.from_preset(preset_name)
        preset_utils.save_to_preset(model, save_dir)

        # Check existence of files
        self.assertTrue(os.path.exists(os.path.join(save_dir, "config.json")))
        self.assertTrue(
            os.path.exists(os.path.join(save_dir, "model.weights.h5"))
        )
        self.assertTrue(os.path.exists(os.path.join(save_dir, "metadata.json")))

        # Check the model config (`config.json`)
        with open(os.path.join(save_dir, "config.json"), "r") as f:
            config_json = f.read()
        self.assertTrue(
            "build_config" not in config_json
        )  # Test on raw json to include nested keys
        self.assertTrue(
            "compile_config" not in config_json
        )  # Test on raw json to include nested keys
        config = json.loads(config_json)
        self.assertEqual(config["weights"], "model.weights.h5")

        # Try loading the model from preset directory
        restored_model = preset_utils.load_from_preset(save_dir)

        input_batch = np.ones(shape=(2, 224, 224, 3))
        expected_output = model(input_batch)
        restored_output = restored_model(input_batch)
        self.assertAllClose(expected_output, restored_output)