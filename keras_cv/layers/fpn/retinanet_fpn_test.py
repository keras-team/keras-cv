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

from keras_cv.layers import RetinaNetFPN


class RetinaNetFPNTest(tf.test.TestCase):
    def test_return_type(self):
        layer = RetinaNetFPN(pyramid_levels=[3, 4, 5, 6, 7])
        C3 = tf.keras.layers.Input([32, 32, 3])
        C4 = tf.keras.layers.Input([16, 16, 3])
        C5 = tf.keras.layers.Input([8, 8, 3])

        xs = layer([C3, C4, C5])
        self.assertTrue(isinstance(xs, dict))

    def test_return_shape(self):
        layer = RetinaNetFPN()
        C3 = tf.ones((2, 64, 64, 3))
        C4 = tf.ones((2, 32, 32, 3))
        C5 = tf.ones((2, 16, 16, 3))

        xs = layer([C3, C4, C5])

        self.assertEqual(xs["P3"].shape, (2, 64, 64, 256))
        self.assertEqual(xs["P4"].shape, (2, 32, 32, 256))
        self.assertEqual(xs["P5"].shape, (2, 16, 16, 256))
        self.assertEqual(xs["P6"].shape, (2, 8, 8, 256))
        self.assertEqual(xs["P7"].shape, (2, 4, 4, 256))

    def test_non_square_images(self):
        layer = RetinaNetFPN(pyramid_levels=[3, 4, 5, 6, 7])
        C3 = tf.ones((2, 64, 128, 3))
        C4 = tf.ones((2, 32, 64, 3))
        C5 = tf.ones((2, 16, 32, 3))

        xs = layer([C3, C4, C5])

        self.assertEqual(xs["P3"].shape, (2, 64, 128, 256))
        self.assertEqual(xs["P4"].shape, (2, 32, 64, 256))
        self.assertEqual(xs["P5"].shape, (2, 16, 32, 256))
        self.assertEqual(xs["P6"].shape, (2, 8, 16, 256))
        self.assertEqual(xs["P7"].shape, (2, 4, 8, 256))

    def test_different_channels(self):
        layer = RetinaNetFPN(pyramid_levels=[3, 4, 5, 6, 7])

        C3 = tf.ones((2, 64, 64, 256))
        C4 = tf.ones((2, 32, 32, 512))
        C5 = tf.ones((2, 16, 16, 1024))

        xs = layer([C3, C4, C5])

        self.assertEqual(xs["P3"].shape, (2, 64, 64, 256))
        self.assertEqual(xs["P4"].shape, (2, 32, 32, 256))
        self.assertEqual(xs["P5"].shape, (2, 16, 16, 256))
        self.assertEqual(xs["P6"].shape, (2, 8, 8, 256))
        self.assertEqual(xs["P7"].shape, (2, 4, 4, 256))

    def test_in_a_model(self):
        layer = RetinaNetFPN(pyramid_levels=[3, 4, 5, 6, 7])

        C3 = tf.keras.layers.Input([64, 64, 3])
        C4 = tf.keras.layers.Input([32, 32, 3])
        C5 = tf.keras.layers.Input([16, 16, 3])
        xs = layer([C3, C4, C5])

        model = tf.keras.models.Model(inputs=[C3, C4, C5], outputs=xs)

        C3 = tf.ones((2, 64, 64, 3))
        C4 = tf.ones((2, 32, 32, 3))
        C5 = tf.ones((2, 16, 16, 3))

        xs = model([C3, C4, C5])

        self.assertEqual(xs["P3"].shape, (2, 64, 64, 256))
        self.assertEqual(xs["P4"].shape, (2, 32, 32, 256))
        self.assertEqual(xs["P5"].shape, (2, 16, 16, 256))
        self.assertEqual(xs["P6"].shape, (2, 8, 8, 256))
        self.assertEqual(xs["P7"].shape, (2, 4, 4, 256))

    def test_adding_level7_without_level6(self):
        with self.assertRaises(ValueError):
            RetinaNetFPN(pyramid_levels=[3, 4, 5, 7])

    def test_skipping_level345(self):
        for invalid_levels in [[3, 5, 6, 7], [3, 4, 6, 7], [4, 5, 6, 7]]:
            with self.assertRaises(ValueError):
                RetinaNetFPN(pyramid_levels=invalid_levels)
