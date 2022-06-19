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

from keras_cv.layers import FeaturePyramid


class FeaturePyramidTest(tf.test.TestCase):
    def test_return_type(self):
        layer = FeaturePyramid(pyramid_levels=[2, 3, 4, 5])
        C2 = tf.keras.layers.Input([64, 64, 3])
        C3 = tf.keras.layers.Input([32, 32, 3])
        C4 = tf.keras.layers.Input([16, 16, 3])
        C5 = tf.keras.layers.Input([8, 8, 3])

        xs = layer([C2, C3, C4, C5])
        self.assertTrue(isinstance(xs, dict))

    def test_return_shape(self):
        layer = FeaturePyramid(pyramid_levels=[2, 3, 4, 5])
        C2 = tf.ones((2, 64, 64, 3))
        C3 = tf.ones((2, 32, 32, 3))
        C4 = tf.ones((2, 16, 16, 3))
        C5 = tf.ones((2, 8, 8, 3))

        xs = layer([C2, C3, C4, C5])

        self.assertEqual(xs["P2"].shape, (2, 64, 64, 256))
        self.assertEqual(xs["P3"].shape, (2, 32, 32, 256))
        self.assertEqual(xs["P4"].shape, (2, 16, 16, 256))
        self.assertEqual(xs["P5"].shape, (2, 8, 8, 256))

    def test_non_square_images(self):
        layer = FeaturePyramid(pyramid_levels=[2, 3, 4, 5])
        C2 = tf.ones((2, 64, 128, 3))
        C3 = tf.ones((2, 32, 64, 3))
        C4 = tf.ones((2, 16, 32, 3))
        C5 = tf.ones((2, 8, 16, 3))

        xs = layer([C2, C3, C4, C5])

        self.assertEqual(xs["P2"].shape, (2, 64, 128, 256))
        self.assertEqual(xs["P3"].shape, (2, 32, 64, 256))
        self.assertEqual(xs["P4"].shape, (2, 16, 32, 256))
        self.assertEqual(xs["P5"].shape, (2, 8, 16, 256))

    def test_different_channels(self):
        layer = FeaturePyramid(pyramid_levels=[2, 3, 4, 5])

        C2 = tf.ones((2, 64, 64, 256))
        C3 = tf.ones((2, 32, 32, 512))
        C4 = tf.ones((2, 16, 16, 1024))
        C5 = tf.ones((2, 8, 8, 2048))

        xs = layer([C2, C3, C4, C5])

        self.assertEqual(xs["P2"].shape, (2, 64, 64, 256))
        self.assertEqual(xs["P3"].shape, (2, 32, 32, 256))
        self.assertEqual(xs["P4"].shape, (2, 16, 16, 256))
        self.assertEqual(xs["P5"].shape, (2, 8, 8, 256))

    def test_in_a_model(self):
        layer = FeaturePyramid(pyramid_levels=[2, 3, 4, 5])

        C2 = tf.keras.layers.Input([64, 64, 3])
        C3 = tf.keras.layers.Input([32, 32, 3])
        C4 = tf.keras.layers.Input([16, 16, 3])
        C5 = tf.keras.layers.Input([8, 8, 3])
        xs = layer([C2, C3, C4, C5])

        model = tf.keras.models.Model(inputs=[C2, C3, C4, C5], outputs=xs)

        C2 = tf.ones((2, 64, 64, 3))
        C3 = tf.ones((2, 32, 32, 3))
        C4 = tf.ones((2, 16, 16, 3))
        C5 = tf.ones((2, 8, 8, 3))

        xs = model([C2, C3, C4, C5])

        self.assertEqual(xs["P2"].shape, (2, 64, 64, 256))
        self.assertEqual(xs["P3"].shape, (2, 32, 32, 256))
        self.assertEqual(xs["P4"].shape, (2, 16, 16, 256))
        self.assertEqual(xs["P5"].shape, (2, 8, 8, 256))
