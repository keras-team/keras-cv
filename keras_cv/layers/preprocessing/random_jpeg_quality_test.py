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

from keras_cv.layers import preprocessing


class RandomJpegQualityTest(tf.test.TestCase):
    def test_return_shapes(self):
        layer = preprocessing.RandomJpegQuality(factor=[0, 100])

        # RGB
        xs = tf.ones((2, 512, 512, 3))
        xs = layer(xs)
        self.assertEqual(xs.shape, [2, 512, 512, 3])

        # greyscale
        xs = tf.ones((2, 512, 512, 1))
        xs = layer(xs)
        self.assertEqual(xs.shape, [2, 512, 512, 1])

    def test_in_single_image(self):
        layer = preprocessing.RandomJpegQuality(factor=[0, 100])

        # RGB
        xs = tf.cast(
            tf.ones((512, 512, 3)),
            dtype=tf.float32,
        )

        xs = layer(xs)
        self.assertEqual(xs.shape, [512, 512, 3])

        # greyscale
        xs = tf.cast(
            tf.ones((512, 512, 1)),
            dtype=tf.float32,
        )

        xs = layer(xs)
        self.assertEqual(xs.shape, [512, 512, 1])

    def test_non_square_images(self):
        layer = preprocessing.RandomJpegQuality(factor=[0, 100])

        # RGB
        xs = tf.ones((2, 256, 512, 3))
        xs = layer(xs)
        self.assertEqual(xs.shape, [2, 256, 512, 3])

        # greyscale
        xs = tf.ones((2, 256, 512, 1))
        xs = layer(xs)
        self.assertEqual(xs.shape, [2, 256, 512, 1])

    def test_augment_bounding_box_dict_input(self):
        input_image = tf.random.uniform((2, 512, 512, 3), 0, 255, dtype=tf.float32)
        bounding_boxes = tf.convert_to_tensor(
            [[200, 200, 400, 400], [100, 100, 300, 300]]
        )
        input = {"images": input_image, "bounding_boxes": bounding_boxes}
        layer = preprocessing.RandomHue(factor=(0.3, 0.8), value_range=(0, 255))
        output = layer(input)
        self.assertAllClose(bounding_boxes, output["bounding_boxes"])
