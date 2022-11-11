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

import tensorflow as tf

from keras_cv.layers.vit_layers import PatchEmbedding
from keras_cv.layers.vit_layers import Patching


class ViTLayersTest(tf.test.TestCase):
    def test_patching_return_type_and_shape(self):
        layer = Patching(patch_size=16)

        inputs = tf.random.normal([1, 224, 224, 3])
        output = layer(inputs)
        self.assertTrue(isinstance(output, tf.Tensor))
        self.assertLen(output, 1)
        self.assertEquals(output.shape, [1, 196, 768])

    def test_patching_wrong_patch_size(self):
        layer = Patching(patch_size=-16)
        inputs = tf.random.normal([1, 224, 224, 3])
        with self.assertRaisesRegexp(
            ValueError,
            "The patch_size cannot be a negative number. Received -16",
        ):
            layer(inputs)

    def test_patching_wrong_padding(self):
        layer = Patching(patch_size=16, padding="REFLECT")
        inputs = tf.random.normal([1, 224, 224, 3])
        with self.assertRaisesRegexp(
            ValueError,
            "Padding must be either 'SAME' or 'VALID', but REFLECT was passed.",
        ):
            layer(inputs)

    def test_patch_embedding_return_type_and_shape(self):
        layer = PatchEmbedding(project_dim=128)
        inputs = tf.random.normal([1, 196, 768])
        output = layer(inputs)
        self.assertTrue(isinstance(output, tf.Tensor))
        self.assertLen(output, 1)
        self.assertEquals(output.shape, [1, 197, 128])

    def test_patch_embedding_interpolation(self):
        inputs = tf.random.normal([1, 196, 768])
        layer = PatchEmbedding(project_dim=128)

        output = layer(
            patch=inputs,
            interpolate=True,
            interpolate_width=300,
            interpolate_height=300,
            patch_size=12,
        )

        self.assertTrue(isinstance(output, tf.Tensor))
        self.assertLen(output, 1)
        self.assertEquals(output.shape, [1, 626, 128])
