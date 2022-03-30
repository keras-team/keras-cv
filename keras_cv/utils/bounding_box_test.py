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

from keras_cv.utils import bounding_box


class BBOXTestCase(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.corner_bounding_box = tf.constant(
            [[10, 10, 110, 110], [20, 20, 120, 120]], dtype=tf.float32
        )
        self.xywh_bounding_box = tf.constant(
            [[60, 60, 100, 100], [70, 70, 100, 100]], dtype=tf.float32
        )

    def test_corner_to_xywh(self):
        self.assertAllClose(
            bounding_box.corners_to_xywh(self.corner_bounding_box),
            self.xywh_bounding_box,
        )

        # Make sure it also accept higher rank than 2
        corner_bounding_box_3d = tf.expand_dims(self.corner_bounding_box, 0)
        xywh_bounding_box_3d = tf.expand_dims(self.xywh_bounding_box, 0)
        self.assertAllClose(
            bounding_box.corners_to_xywh(corner_bounding_box_3d), xywh_bounding_box_3d
        )

        # Make sure it also accept more value after last index.
        padded_corner_bounding_box = tf.pad(
            self.corner_bounding_box, [[0, 0], [0, 2]]
        )  # Right pad 2 more value
        padded_xywh_bounding_box = tf.pad(self.xywh_bounding_box, [[0, 0], [0, 2]])
        self.assertAllClose(
            bounding_box.corners_to_xywh(padded_corner_bounding_box),
            padded_xywh_bounding_box,
        )

        # Same for higher rank
        padded_corner_bounding_box_3d = tf.expand_dims(padded_corner_bounding_box, 0)
        padded_xywh_bounding_box_3d = tf.expand_dims(padded_xywh_bounding_box, 0)
        self.assertAllClose(
            bounding_box.corners_to_xywh(padded_corner_bounding_box_3d),
            padded_xywh_bounding_box_3d,
        )

    def test_xywh_to_corner(self):
        self.assertAllClose(
            bounding_box.xywh_to_corners(self.xywh_bounding_box),
            self.corner_bounding_box,
        )

        # Make sure it also accept higher rank than 2
        corner_bounding_box_3d = tf.expand_dims(self.corner_bounding_box, 0)
        xywh_bounding_box_3d = tf.expand_dims(self.xywh_bounding_box, 0)
        self.assertAllClose(
            bounding_box.xywh_to_corners(xywh_bounding_box_3d), corner_bounding_box_3d
        )

        # Make sure it also accept more value after last index.
        padded_corner_bounding_box = tf.pad(
            self.corner_bounding_box, [[0, 0], [0, 2]]
        )  # Right pad 2 more value
        padded_xywh_bounding_box = tf.pad(self.xywh_bounding_box, [[0, 0], [0, 2]])
        self.assertAllClose(
            bounding_box.xywh_to_corners(padded_xywh_bounding_box),
            padded_corner_bounding_box,
        )

        # Same for higher rank
        padded_corner_bounding_box_3d = tf.expand_dims(padded_corner_bounding_box, 0)
        padded_xywh_bounding_box_3d = tf.expand_dims(padded_xywh_bounding_box, 0)
        self.assertAllClose(
            bounding_box.xywh_to_corners(padded_xywh_bounding_box_3d),
            padded_corner_bounding_box_3d,
        )

    def test_bounding_box_padding(self):
        bounding_boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]
        target_shape = [3, 4]
        result = bounding_box.pad_bounding_box_batch_to_shape(
            bounding_boxes, target_shape
        )
        self.assertAllClose(result, [[1, 2, 3, 4], [5, 6, 7, 8], [-1, -1, -1, -1]])

        target_shape = [2, 5]
        result = bounding_box.pad_bounding_box_batch_to_shape(
            bounding_boxes, target_shape
        )
        self.assertAllClose(result, [[1, 2, 3, 4, -1], [5, 6, 7, 8, -1]])

        # Make sure to raise error if the rank is different between bounding_box and
        # target shape
        with self.assertRaisesRegex(ValueError, "Target shape should have same rank"):
            bounding_box.pad_bounding_box_batch_to_shape(bounding_boxes, [1, 2, 3])

        # Make sure raise error if the target shape is smaller
        target_shape = [3, 2]
        with self.assertRaisesRegex(
            ValueError, "Target shape should be larger than bounding box shape"
        ):
            bounding_box.pad_bounding_box_batch_to_shape(bounding_boxes, target_shape)
