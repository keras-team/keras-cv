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

import numpy as np
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
        with self.assertRaisesRegexp(
            ValueError,
            "Padding must be either 'SAME' or 'VALID', but REFLECT was passed.",
        ):
            Patching(patch_size=16, padding="REFLECT")

    def test_patch_embedding_return_type_and_shape(self):
        layer = PatchEmbedding(project_dim=128)
        inputs = tf.random.normal([1, 196, 768])
        output = layer(inputs)
        self.assertTrue(isinstance(output, tf.Tensor))
        self.assertLen(output, 1)
        self.assertEquals(output.shape, [1, 197, 128])

    def test_patch_embedding_interpolation(self):
        inputs = tf.ones([1, 625, 432])
        patch_embedding = PatchEmbedding(project_dim=128)
        patch_embedding.build(inputs.shape)

        positional_embeddings = tf.ones([626, 128])
        (
            output,
            cls,
        ) = patch_embedding._PatchEmbedding__interpolate_positional_embeddings(
            positional_embeddings, height=450, width=450, patch_size=12
        )

        self.assertTrue(isinstance(output, tf.Tensor))
        self.assertLen(output, 1)
        self.assertEquals(output.shape, [1, 1369, 128])

    def test_patch_embedding_interpolation_numerical(self):
        inputs = tf.ones([1, 16, 3])
        patch_embedding = PatchEmbedding(project_dim=4)
        patch_embedding.build(inputs.shape)

        positional_embeddings = tf.ones([17, 4])
        (
            output,
            cls_token,
        ) = patch_embedding._PatchEmbedding__interpolate_positional_embeddings(
            positional_embeddings, height=8, width=8, patch_size=2
        )

        self.assertTrue(tf.reduce_all(tf.equal(output, tf.ones([1, 16, 4]))).numpy())

    def test_patching_numerical(self):
        layer = Patching(patch_size=1)
        input_img = np.array(
            [
                [
                    [[1.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
                    [[70.0, 80.0, 90.0], [100.0, 110.0, 120.0]],
                ]
            ]
        )

        input_img = tf.convert_to_tensor(input_img)
        output = layer(input_img)

        expected_output = np.array(
            [
                [
                    [1.0, 20.0, 30.0],
                    [40.0, 50.0, 60.0],
                    [70.0, 80.0, 90.0],
                    [100.0, 110.0, 120.0],
                ]
            ]
        )

        expected_output = tf.convert_to_tensor(expected_output, dtype=tf.float32)
        self.assertTrue(tf.reduce_all(tf.equal(output, expected_output)).numpy())
