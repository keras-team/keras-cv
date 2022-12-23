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
        layer = NonMaxSuppression(classes=4, bounding_box_format="xyWH")
        images = tf.ones((3, 480, 480, 3))

        boxes = tf.cast(tf.random.uniform((3, 5, 4), 0, 480, tf.int32), tf.float32)
        classes = tf.cast(tf.random.uniform((3, 5), 0, 4, tf.int32), tf.float32)
        scores = tf.random.uniform((3, 5), 0, 1, tf.float32)

        predictions = {"boxes": boxes, "classes": classes, "confidence": scores}

        outputs = layer(predictions, images)
        self.assertEqual(outputs["boxes"].shape, [3, 100, 4])
        self.assertEqual(outputs["classes"].shape, [3, 100])
        self.assertEqual(outputs["confidence"].shape, [3, 100])

    def test_non_square_images(self):
        layer = NonMaxSuppression(classes=4, bounding_box_format="xyxy")

        boxes = tf.cast(tf.random.uniform((3, 5, 4), 0, 480, tf.int32), tf.float32)
        classes = tf.cast(tf.random.uniform((3, 5), 0, 4, tf.int32), tf.float32)
        scores = tf.random.uniform((3, 5), 0, 1, tf.float32)
        predictions = {"boxes": boxes, "classes": classes, "confidence": scores}

        # RGB image
        images = tf.ones((2, 256, 512, 3))
        outputs = layer(predictions, images)
        self.assertEqual(outputs["boxes"].shape, [3, 100, 4])
        self.assertEqual(outputs["classes"].shape, [3, 100])
        self.assertEqual(outputs["confidence"].shape, [3, 100])

        # grayscale image
        images = tf.ones((2, 256, 512, 1))
        outputs = layer(predictions, images)
        self.assertEqual(outputs["boxes"].shape, [3, 100, 4])
        self.assertEqual(outputs["classes"].shape, [3, 100])
        self.assertEqual(outputs["confidence"].shape, [3, 100])

    def test_without_images(self):
        layer = NonMaxSuppression(classes=4, bounding_box_format="xyWH")

        boxes = tf.cast(tf.random.uniform((3, 5, 4), 0, 480, tf.int32), tf.float32)
        classes = tf.cast(tf.random.uniform((3, 5), 0, 4, tf.int32), tf.float32)
        scores = tf.random.uniform((3, 5), 0, 1, tf.float32)
        predictions = {"boxes": boxes, "classes": classes, "confidence": scores}

        outputs = layer(predictions)
        self.assertEqual(outputs["boxes"].shape, [3, 100, 4])
        self.assertEqual(outputs["classes"].shape, [3, 100])
        self.assertEqual(outputs["confidence"].shape, [3, 100])

    def test_ragged_output_with_differing_shapes(self):
        layer = NonMaxSuppression(8, "xywh", iou_threshold=0.1)
        images = tf.ones((2, 480, 480, 3))

        boxes = [
            [
                [
                    0,
                    0,
                    1,
                    1,
                ],
                [
                    0,
                    0,
                    2,
                    3,
                ],
                [
                    4,
                    5,
                    3,
                    6,
                ],
                [
                    2,
                    2,
                    3,
                    3,
                ],
            ],
            [
                [
                    0,
                    0,
                    5,
                    6,
                ],
                [
                    0,
                    0,
                    7,
                    3,
                ],
                [
                    4,
                    5,
                    5,
                    6,
                ],
                [
                    2,
                    1,
                    3,
                    3,
                ],
            ],
        ]
        classes = [[4, 4, 3, 6], [4, 1, 4, 7]]
        confidence = [[0.9, 0.76, 0.89, 0.04], [0.9, 0.76, 0.04, 0.48]]
        predictions = {
            "boxes": tf.convert_to_tensor(
                boxes,
                dtype=tf.float32,
            ),
            "classes": tf.convert_to_tensor(classes, tf.float32),
            "confidence": tf.convert_to_tensor(confidence, tf.float32),
        }

        outputs = layer(predictions, images)
        self.assertEqual(outputs["boxes"][0].shape, [2, 4])
        self.assertEqual(outputs["classes"][0].shape, [2])
        self.assertEqual(outputs["confidence"][0].shape, [2])
        self.assertEqual(outputs["boxes"][1].shape, [3, 4])
        self.assertEqual(outputs["classes"][1].shape, [3])
        self.assertEqual(outputs["confidence"][1].shape, [3])

    def test_ragged_output_with_zero_boxes(self):
        layer = NonMaxSuppression(8, "xywh", confidence_threshold=0.1)
        images = tf.ones((2, 480, 480, 3))

        boxes = tf.cast(tf.random.uniform((3, 5, 4), 0, 480, tf.int32), tf.float32)
        classes = tf.cast(tf.random.uniform((3, 5), 0, 4, tf.int32), tf.float32)
        scores = tf.random.uniform((3, 5), 0, 0.01, tf.float32)
        predictions = {"boxes": boxes, "classes": classes, "confidence": scores}

        result = layer(predictions, images)
        self.assertEqual(result["boxes"][0].shape, [0, 4])
        self.assertEqual(result["boxes"][1].shape, [0, 4])
