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

from keras_cv.backend import ops
from keras_cv.layers.detectron2_layers import AddPositionalEmbedding
from keras_cv.layers.detectron2_layers import MultiHeadAttentionWithRelativePE
from keras_cv.layers.detectron2_layers import ViTDetPatchingAndEmbedding
from keras_cv.layers.detectron2_layers import WindowedTransformerEncoder
from keras_cv.tests.test_case import TestCase


class TestDetectron2Layers(TestCase):
    def test_multi_head_attention_with_relative_pe(self):
        attention_with_rel_pe = MultiHeadAttentionWithRelativePE(
            num_heads=16,
            key_dim=1280 // 16,
            use_bias=True,
            input_size=(64, 64),
        )
        x = np.ones(shape=(1, 64, 64, 1280))
        x_out = ops.convert_to_numpy(attention_with_rel_pe(x))
        self.assertEqual(x_out.shape, (1, 64, 64, 1280))

    def test_windowed_transformer_encoder(self):
        windowed_transformer_encoder = WindowedTransformerEncoder(
            project_dim=1280,
            mlp_dim=1280 * 4,
            num_heads=16,
            use_bias=True,
            use_rel_pos=True,
            window_size=14,
            input_size=(64, 64),
        )
        x = np.ones((1, 64, 64, 1280))
        x_out = ops.convert_to_numpy(windowed_transformer_encoder(x))
        self.assertEqual(x_out.shape, (1, 64, 64, 1280))
        self.assertAllClose(x_out, np.ones_like(x_out))

    def test_vit_patching_and_embedding(self):
        vit_patching_and_embedding = ViTDetPatchingAndEmbedding()
        x = np.ones((1, 1024, 1024, 3))
        x_out = vit_patching_and_embedding(x)
        self.assertEqual(x_out.shape, (1, 64, 64, 768))

    def test_add_positional_embedding(self):
        add_positional_embedding = AddPositionalEmbedding(
            img_size=1024, patch_size=16, embed_dim=256
        )
        x = np.ones((1, 64, 64, 256))
        x_out = add_positional_embedding(x)
        self.assertEqual(x_out.shape, (1, 64, 64, 256))
