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

from keras_cv.layers import NonMaxSuppression


class NonMaxSuppressionTest(tf.test.TestCase):
    def test_return_shapes(self):
        layer = NonMaxSuppression(num_classes=4, bounding_box_format="xyWH")
        images = tf.ones((3, 480, 480, 3))

        boxes = tf.cast(tf.random.uniform((3, 5, 4), 0, 480, tf.int32), tf.float32)
        classes = tf.cast(tf.random.uniform((3, 5, 1), 0, 4, tf.int32), tf.float32)
        scores = tf.random.uniform((3, 5, 1), 0, 1, tf.float32)

        predictions = tf.concat([boxes, classes, scores], axis=-1)

        boxes = layer(predictions, images)
        self.assertEqual(boxes.shape, [3, None, 6])

    def test_non_square_images(self):
        layer = NonMaxSuppression(num_classes=4, bounding_box_format="xyxy")

        boxes = tf.cast(tf.random.uniform((2, 5, 4), 0, 480, tf.int32), tf.float32)
        classes = tf.cast(tf.random.uniform((2, 5, 1), 0, 4, tf.int32), tf.float32)
        scores = tf.random.uniform((2, 5, 1), 0, 1, tf.float32)

        predictions = tf.concat([boxes, classes, scores], axis=-1)

        # RGB
        images = tf.ones((2, 256, 512, 3))
        boxes = layer(predictions, images)
        self.assertEqual(boxes.shape, [2, None, 6])

        # greyscale
        images = tf.ones((2, 256, 512, 1))
        boxes = layer(predictions, images)
        self.assertEqual(boxes.shape, [2, None, 6])

    def test_different_channels(self):
        layer = NonMaxSuppression(num_classes=4, bounding_box_format="xyWH")
        images = tf.ones((3, 480, 480, 5))

        boxes = tf.cast(tf.random.uniform((3, 5, 4), 0, 480, tf.int32), tf.float32)
        classes = tf.cast(tf.random.uniform((3, 5, 1), 0, 4, tf.int32), tf.float32)
        scores = tf.random.uniform((3, 5, 1), 0, 1, tf.float32)

        predictions = tf.concat([boxes, classes, scores], axis=-1)

        boxes = layer(predictions, images)
        self.assertEqual(boxes.shape, [3, None, 6])

    def test_in_a_model(self):
        input1 = tf.keras.layers.Input([5, 6])
        input2 = tf.keras.layers.Input([480, 480, 3])
        layer = NonMaxSuppression(num_classes=4, bounding_box_format="xyWH")
        outputs = layer(input1, input2)

        model = tf.keras.models.Model(inputs=[input1, input2], outputs=outputs)

        images = tf.ones((3, 480, 480, 3))

        boxes = tf.cast(tf.random.uniform((3, 5, 4), 0, 480, tf.int32), tf.float32)
        classes = tf.cast(tf.random.uniform((3, 5, 1), 0, 4, tf.int32), tf.float32)
        scores = tf.random.uniform((3, 5, 1), 0, 1, tf.float32)

        predictions = tf.concat([boxes, classes, scores], axis=-1)

        boxes = model([predictions, images])
        self.assertEqual(boxes.shape, [3, None, 6])

    def test_without_images(self):
        layer = NonMaxSuppression(num_classes=4, bounding_box_format="xyWH")

        boxes = tf.cast(tf.random.uniform((3, 5, 4), 0, 480, tf.int32), tf.float32)
        classes = tf.cast(tf.random.uniform((3, 5, 1), 0, 4, tf.int32), tf.float32)
        scores = tf.random.uniform((3, 5, 1), 0, 1, tf.float32)

        predictions = tf.concat([boxes, classes, scores], axis=-1)

        boxes = layer(predictions)
        self.assertEqual(boxes.shape, [3, None, 6])
