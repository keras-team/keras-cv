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

import tensorflow as tf

from keras_cv.models.object_detection.yolox.layers import YoloXLabelEncoder
from keras_cv.tests.test_case import TestCase


class YoloXLabelEncoderTest(TestCase):
    def test_ragged_images_exception(self):
        img1 = tf.random.uniform((10, 11, 3))
        img2 = tf.random.uniform((9, 14, 3))
        img3 = tf.random.uniform((7, 12, 3))

        images = tf.ragged.stack([img1, img2, img3])
        box_labels = {}
        box_labels["bounding_boxes"] = tf.random.uniform((3, 4, 4))
        box_labels["classes"] = tf.random.uniform(
            (3, 4), maxval=20, dtype=tf.int32
        )
        layer = YoloXLabelEncoder()

        with self.assertRaisesRegexp(
            ValueError,
            "method does not support RaggedTensor inputs for the `images` "
            "argument.",
        ):
            layer(images, box_labels)

    def test_ragged_labels(self):
        images = tf.random.uniform((3, 12, 12, 3))

        box_labels = {}

        box1 = tf.random.uniform((11, 4))
        class1 = tf.random.uniform([11], maxval=20, dtype=tf.int32)
        box2 = tf.random.uniform((14, 4))
        class2 = tf.random.uniform([14], maxval=20, dtype=tf.int32)
        box3 = tf.random.uniform((12, 4))
        class3 = tf.random.uniform([12], maxval=20, dtype=tf.int32)

        box_labels["boxes"] = tf.ragged.stack([box1, box2, box3])
        box_labels["classes"] = tf.ragged.stack([class1, class2, class3])

        layer = YoloXLabelEncoder()

        encoded_boxes, _ = layer(images, box_labels)
        self.assertEqual(encoded_boxes.shape, (3, 14, 4))

    def test_one_hot_classes_exception(self):
        images = tf.random.uniform((3, 12, 12, 3))

        box_labels = {}

        box1 = tf.random.uniform((11, 4))
        class1 = tf.random.uniform([11], maxval=20, dtype=tf.int32)
        class1 = tf.one_hot(class1, 20)

        box2 = tf.random.uniform((14, 4))
        class2 = tf.random.uniform([14], maxval=20, dtype=tf.int32)
        class2 = tf.one_hot(class2, 20)

        box3 = tf.random.uniform((12, 4))
        class3 = tf.random.uniform([12], maxval=20, dtype=tf.int32)
        class3 = tf.one_hot(class3, 20)

        box_labels["boxes"] = tf.ragged.stack([box1, box2, box3])
        box_labels["classes"] = tf.ragged.stack([class1, class2, class3])

        layer = YoloXLabelEncoder()

        with self.assertRaises(ValueError):
            layer(images, box_labels)
