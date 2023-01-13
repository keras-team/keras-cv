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
from absl.testing import parameterized

from keras_cv import bounding_box
from keras_cv.layers import preprocessing


class CenterCropTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.product(target_height=[5, 10, 15, 20], target_width=[5, 10, 15, 20])
    def test_same_results(self, target_height, target_width):
        images = tf.random.normal((2, 10, 10, 3))

        original_layer = tf.keras.layers.CenterCrop(target_height, target_width)
        layer = preprocessing.CenterCrop(target_height, target_width)

        original_output_image = original_layer(images)
        output = layer({"images": images})

        self.assertShapeEqual(original_output_image, output["images"])
        self.assertAllClose(original_output_image, output["images"])

    def test_bounding_boxes_case_crop(self):
        images = tf.random.normal([2, 20, 20, 3])
        boxes = tf.constant(
            [
                [[0, 0, 15, 15], [0, 0, 10, 10], [0, 0, 5, 5], [18, 18, 20, 20]],
                [[0, 0, 15, 15], [0, 0, 10, 10], [0, 0, 5, 5], [18, 18, 20, 20]],
            ],
            dtype=tf.float32
        )
        classes = tf.convert_to_tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        bboxes = {
            "boxes": boxes,
            "classes": classes
        }

        layer = preprocessing.CenterCrop(10, 10)
        layer.bounding_box_format = "xyxy"

        outputs = layer({"images": images, "bounding_boxes": bboxes})

        expected_boxes = tf.ragged.constant(
            [
                [0, 0, 10, 10],
                [0, 0, 5, 5]
            ], dtype=tf.float32
        )
        expected_classes = tf.convert_to_tensor(
            [0, 0], dtype=tf.int32
        )

        self.assertAllClose(outputs["images"], images[:, 5:15, 5:15, :])
        self.assertAllClose(outputs["bounding_boxes"]["boxes"][0], expected_boxes)
        self.assertAllClose(outputs["bounding_boxes"]["boxes"][1], expected_boxes)
        self.assertAllClose(outputs["bounding_boxes"]["classes"][0], expected_classes)
        self.assertAllClose(outputs["bounding_boxes"]["classes"][1], expected_classes)

    def test_bounding_boxes_upsample1(self):
        images = tf.random.normal([2, 20, 20, 3])
        boxes = tf.constant(
            [
                [[0, 0, 15, 15], [0, 0, 10, 10], [0, 0, 5, 5], [18, 18, 20, 20]],
                [[0, 0, 15, 15], [0, 0, 10, 10], [0, 0, 5, 5], [18, 18, 20, 20]],
            ],
            dtype=tf.float32,
        )
        classes = tf.convert_to_tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        bboxes = {
            "boxes": boxes,
            "classes": classes
        }

        layer = preprocessing.CenterCrop(10, 30)
        layer.bounding_box_format = "xyxy"

        outputs = layer({"images": images, "bounding_boxes": bboxes})

        expected_bboxes = tf.ragged.constant(
            [
                [0, 0, 22.5, 10],
                [0, 0, 15, 5]
            ], dtype=tf.float32
        )

        self.assertAllClose(outputs["bounding_boxes"]["boxes"][0], expected_bboxes)
        self.assertAllClose(outputs["bounding_boxes"]["boxes"][1], expected_bboxes)

    def test_bounding_boxes_upsample2(self):
        images = tf.random.normal([2, 20, 20, 3])
        boxes = tf.constant(
            [
                [[0, 0, 15, 15], [0, 0, 10, 10], [0, 0, 5, 5], [18, 18, 20, 20]],
                [[0, 0, 15, 15], [0, 0, 10, 10], [0, 0, 5, 5], [18, 18, 20, 20]],
            ],
            dtype=tf.float32,
        )
        classes = tf.convert_to_tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        bboxes = {
            "boxes": boxes,
            "classes": classes
        }

        layer = preprocessing.CenterCrop(25, 50)
        layer.bounding_box_format = "xyxy"

        outputs = layer({"images": images, "bounding_boxes": bboxes})

        expected_bboxes = tf.ragged.constant(
            [
                [0, 0, 37.5, 25],
                [0, 0, 25, 12.5]
            ], dtype=tf.float32
        )

        self.assertAllClose(outputs["bounding_boxes"]["boxes"][0], expected_bboxes)
        self.assertAllClose(outputs["bounding_boxes"]["boxes"][1], expected_bboxes)
