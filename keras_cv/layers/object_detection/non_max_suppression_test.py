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

from keras_cv.layers.object_detection.non_max_suppression import NonMaxSuppression


class NonMaxSuppressionTest(tf.test.TestCase):
    def test_return_shapes(self):
        layer = NonMaxSuppression(classes=4, bounding_box_format="xyWH")
        images = tf.ones((3, 480, 480, 3))

        boxes = tf.cast(tf.random.uniform((3, 5, 4), 0, 480, tf.int32), tf.float32)
        classes = tf.cast(tf.random.uniform((3, 5), 0, 4, tf.int32), tf.float32)
        confidence = tf.random.uniform((3, 5), 0, 1, tf.float32)

        predictions = {"boxes": boxes, "classes": classes, "confidence": confidence}

        predictions = layer(predictions, images)
        self.assertEqual(predictions["boxes"].shape, [3, 100, 4])
        self.assertEqual(predictions["classes"].shape, [3, 100])
        self.assertEqual(predictions["confidence"].shape, [3, 100])

    def test_non_square_images(self):
        layer = NonMaxSuppression(classes=4, bounding_box_format="xyxy")

        boxes = tf.cast(tf.random.uniform((3, 5, 4), 0, 480, tf.int32), tf.float32)
        classes = tf.cast(tf.random.uniform((3, 5), 0, 4, tf.int32), tf.float32)
        confidence = tf.random.uniform((3, 5), 0, 1, tf.float32)
        predictions = {"boxes": boxes, "classes": classes, "confidence": confidence}

        # RGB image
        images = tf.ones((2, 256, 512, 3))
        predicts = layer(predictions, images)
        self.assertEqual(predicts["boxes"].shape, [2, 100, 4])
        self.assertEqual(predicts["classes"].shape, [2, 100])
        self.assertEqual(predicts["confidence"].shape, [2, 100])

        # grayscale image
        images = tf.ones((2, 256, 512, 1))
        predicts = layer(predictions, images)
        self.assertEqual(predicts["boxes"].shape, [2, 100, 4])
        self.assertEqual(predicts["classes"].shape, [2, 100])
        self.assertEqual(predicts["confidence"].shape, [2, 100])

    def test_different_channels(self):
        layer = NonMaxSuppression(classes=4, bounding_box_format="xyWH")
        images = tf.ones((3, 480, 480, 5))

        boxes = tf.cast(tf.random.uniform((3, 5, 4), 0, 480, tf.int32), tf.float32)
        classes = tf.cast(tf.random.uniform((3, 5, 1), 0, 4, tf.int32), tf.float32)
        confidence = tf.random.uniform((3, 5, 1), 0, 1, tf.float32)

        predictions = tf.concat([boxes, classes, confidence], axis=-1)

        predictions = layer(predictions, images)
        self.assertEqual(predictions["boxes"].shape, [3, 100, 4])
        self.assertEqual(predictions["classes"].shape, [3, 100])
        self.assertEqual(predictions["confidence"].shape, [3, 100])

    def test_in_a_model(self):
        input1 = tf.keras.layers.Input([5, 6])
        input2 = tf.keras.layers.Input([480, 480, 3])
        layer = NonMaxSuppression(classes=4, bounding_box_format="xyWH")
        outputs = layer(input1, input2)

        model = tf.keras.models.Model(inputs=[input1, input2], outputs=outputs)

        images = tf.ones((3, 480, 480, 3))

        boxes = tf.cast(tf.random.uniform((3, 5, 4), 0, 480, tf.int32), tf.float32)
        classes = tf.cast(tf.random.uniform((3, 5, 1), 0, 4, tf.int32), tf.float32)
        confidence = tf.random.uniform((3, 5, 1), 0, 1, tf.float32)

        predictions = tf.concat([boxes, classes, confidence], axis=-1)

        predictions = model([predictions, images])
        self.assertEqual(predictions["boxes"].shape, [3, 100, 4])
        self.assertEqual(predictions["classes"].shape, [3, 100])
        self.assertEqual(predictions["confidence"].shape, [3, 100])

    def test_without_images(self):
        layer = NonMaxSuppression(classes=4, bounding_box_format="xyWH")

        boxes = tf.cast(tf.random.uniform((3, 5, 4), 0, 480, tf.int32), tf.float32)
        classes = tf.cast(tf.random.uniform((3, 5), 0, 4, tf.int32), tf.float32)
        confidence = tf.random.uniform((3, 5), 0, 1, tf.float32)
        predictions = {"boxes": boxes, "classes": classes, "confidence": confidence}
        predictions = layer(predictions)
        self.assertEqual(predictions["boxes"].shape, [3, 100, 4])
        self.assertEqual(predictions["classes"].shape, [3, 100])
        self.assertEqual(predictions["confidence"].shape, [3, 100])

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

        boxes = layer(predictions, images)
        self.assertEqual(boxes["boxes"].shape, [2, 100, 4])

    def test_ragged_output_with_zero_boxes(self):
        layer = NonMaxSuppression(8, "xywh", confidence_threshold=0.1)
        images = tf.ones((2, 480, 480, 3))

        boxes = tf.cast(tf.random.uniform((3, 5, 4), 0, 480, tf.int32), tf.float32)
        classes = tf.cast(tf.random.uniform((3, 5), 0, 4, tf.int32), tf.float32)
        confidence = tf.random.uniform((3, 5), 0, 0.01, tf.float32)
        predictions = {"boxes": boxes, "classes": classes, "confidence": confidence}

        predictions = tf.concat([boxes, classes, confidence], axis=-1)

        boxes = layer(predictions, images)
        self.assertEqual(boxes["boxes"].shape, [3, 100, 4])

    def test_input_box_shape(self):
        layer = NonMaxSuppression(8, "xywh", confidence_threshold=0.1)
        images = tf.ones((2, 480, 480, 3))

        boxes = tf.cast(tf.random.uniform((3, 5, 5), 0, 480, tf.int32), tf.float32)
        classes = tf.cast(tf.random.uniform((3, 5, 1), 0, 4, tf.int32), tf.float32)
        confidence = tf.random.uniform((3, 5, 1), 0, 0.1, tf.float32)

        predictions = tf.concat([boxes, classes, confidence], axis=-1)

        with self.assertRaises(ValueError):
            boxes = layer(predictions, images)
