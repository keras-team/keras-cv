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

import numpy as np
import pytest

from keras_cv.backend import ops
from keras_cv.layers.video_swin_transformer_layers import PatchEmbedding3D
from keras_cv.layers.video_swin_transformer_layers import WindowAttention3D
from keras_cv.layers.video_swin_transformer_layers import SwinTransformerBlock3D
from keras_cv.tests.test_case import TestCase

class TestPatchEmbedding3D(TestCase):

    def test_patch_embedding_compute_output_shape(self):
        patch_embedding_model = PatchEmbedding3D(patch_size=(2, 4, 4), embed_dim=96, norm_layer=None)
        input_shape = (None, 16, 32, 32, 3)
        output_shape = patch_embedding_model.compute_output_shape(input_shape)
        expected_output_shape = (None, 8, 8, 8, 96)
        self.assertEqual(output_shape, expected_output_shape)
    
    def test_patch_embedding_get_config(self):
        patch_embedding_model = PatchEmbedding3D(patch_size=(4, 4, 4), embed_dim=96)
        config = patch_embedding_model.get_config()
        assert isinstance(config, dict)
        assert config["patch_size"] == (4, 4, 4)
        assert config["embed_dim"] == 96


