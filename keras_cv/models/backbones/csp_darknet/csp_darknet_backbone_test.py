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
from absl.testing import parameterized

from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.models.backbones.csp_darknet.csp_darknet_aliases import (
    CSPDarkNetLBackbone,
)
from keras_cv.models.backbones.csp_darknet.csp_darknet_aliases import (
    CSPDarkNetMBackbone,
)
from keras_cv.models.backbones.csp_darknet.csp_darknet_aliases import (
    CSPDarkNetSBackbone,
)
from keras_cv.models.backbones.csp_darknet.csp_darknet_aliases import (
    CSPDarkNetTinyBackbone,
)
from keras_cv.models.backbones.csp_darknet.csp_darknet_aliases import (
    CSPDarkNetXLBackbone,
)
from keras_cv.models.backbones.csp_darknet.csp_darknet_backbone import (
    CSPDarkNetBackbone,
)
from keras_cv.utils.train import get_feature_extractor


class CSPDarkNetBackboneTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.input_batch = np.ones(shape=(2, 224, 224, 3))

    def test_valid_call(self):
        model = CSPDarkNetBackbone(
            stackwise_channels=[48, 96, 192, 384],
            stackwise_depth=[1, 3, 3, 1],
            include_rescaling=False,
        )
        model(self.input_batch)

    def test_valid_call_applications_model(self):
        model = CSPDarkNetLBackbone()
        model(self.input_batch)

    def test_valid_call_with_rescaling(self):
        model = CSPDarkNetBackbone(
            stackwise_channels=[48, 96, 192, 384],
            stackwise_depth=[1, 3, 3, 1],
            include_rescaling=True,
        )
        model(self.input_batch)

    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self, save_format, filename):
        model = CSPDarkNetBackbone(
            stackwise_channels=[48, 96, 192, 384],
            stackwise_depth=[1, 3, 3, 1],
            include_rescaling=True,
        )
        model_output = model(self.input_batch)
        save_path = os.path.join(
            self.get_temp_dir(), "csp_darknet_backbone.keras"
        )
        model.save(save_path)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, CSPDarkNetBackbone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(
            ops.convert_to_numpy(model_output),
            ops.convert_to_numpy(restored_output),
        )

    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_alias_model(self, save_format, filename):
        model = CSPDarkNetLBackbone()
        model_output = model(self.input_batch)
        save_path = os.path.join(
            self.get_temp_dir(), "csp_darknet_backbone.keras"
        )
        model.save(save_path)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        # Note that these aliases serialized as the base class
        self.assertIsInstance(restored_model, CSPDarkNetBackbone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(
            ops.convert_to_numpy(model_output),
            ops.convert_to_numpy(restored_output),
        )

    def test_feature_pyramid_inputs(self):
        model = CSPDarkNetLBackbone()
        backbone_model = get_feature_extractor(
            model,
            model.pyramid_level_inputs.values(),
            model.pyramid_level_inputs.keys(),
        )
        input_size = 256
        inputs = keras.Input(shape=[input_size, input_size, 3])
        outputs = backbone_model(inputs)
        levels = ["P2", "P3", "P4", "P5"]
        self.assertEquals(list(outputs.keys()), levels)
        self.assertEquals(
            outputs["P2"].shape,
            (None, input_size // 2**2, input_size // 2**2, 128),
        )
        self.assertEquals(
            outputs["P3"].shape,
            (None, input_size // 2**3, input_size // 2**3, 256),
        )
        self.assertEquals(
            outputs["P4"].shape,
            (None, input_size // 2**4, input_size // 2**4, 512),
        )
        self.assertEquals(
            outputs["P5"].shape,
            (None, input_size // 2**5, input_size // 2**5, 1024),
        )

    @parameterized.named_parameters(
        ("Tiny", CSPDarkNetTinyBackbone),
        ("S", CSPDarkNetSBackbone),
        ("M", CSPDarkNetMBackbone),
        ("L", CSPDarkNetLBackbone),
        ("XL", CSPDarkNetXLBackbone),
    )
    def test_specific_arch_forward_pass(self, arch_class):
        backbone = arch_class()
        backbone(np.random.uniform(size=(2, 256, 256, 3)))

    @parameterized.named_parameters(
        ("Tiny", CSPDarkNetTinyBackbone),
        ("S", CSPDarkNetSBackbone),
        ("M", CSPDarkNetMBackbone),
        ("L", CSPDarkNetLBackbone),
        ("XL", CSPDarkNetXLBackbone),
    )
    def test_specific_arch_presets(self, arch_class):
        self.assertDictEqual(
            arch_class.presets, arch_class.presets_with_weights
        )


if __name__ == "__main__":
    tf.test.main()
