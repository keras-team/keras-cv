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

from keras_cv.utils import bbox


class BBOXTestCase(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.corner_bbox = tf.constant(
            [[10, 10, 110, 110], [20, 20, 120, 120]], dtype=tf.float32
        )
        self.xywh_bbox = tf.constant(
            [[60, 60, 100, 100], [70, 70, 100, 100]], dtype=tf.float32
        )

    def test_corner_to_xywh(self):
        self.assertAllClose(bbox.corners_to_xywh(self.corner_bbox), self.xywh_bbox)

        # Make sure it also accept higher rank than 2
        corner_bbox_3d = tf.expand_dims(self.corner_bbox, 0)
        xywh_bbox_3d = tf.expand_dims(self.xywh_bbox, 0)
        self.assertAllClose(bbox.corners_to_xywh(corner_bbox_3d), xywh_bbox_3d)

        # Make sure it also accept more value after last index.
        padded_corner_bbox = tf.pad(
            self.corner_bbox, [[0, 0], [0, 2]]
        )  # Right pad 2 more value
        padded_xywh_bbox = tf.pad(self.xywh_bbox, [[0, 0], [0, 2]])
        self.assertAllClose(
            bbox.corners_to_xywh(padded_corner_bbox), padded_xywh_bbox
        )

        # Same for higher rank
        padded_corner_bbox_3d = tf.expand_dims(padded_corner_bbox, 0)
        padded_xywh_bbox_3d = tf.expand_dims(padded_xywh_bbox, 0)
        self.assertAllClose(
            bbox.corners_to_xywh(padded_corner_bbox_3d), padded_xywh_bbox_3d
        )

    def test_xywh_to_corner(self):
        self.assertAllClose(bbox.xywh_to_corners(self.xywh_bbox), self.corner_bbox)

        # Make sure it also accept higher rank than 2
        corner_bbox_3d = tf.expand_dims(self.corner_bbox, 0)
        xywh_bbox_3d = tf.expand_dims(self.xywh_bbox, 0)
        self.assertAllClose(bbox.xywh_to_corners(xywh_bbox_3d), corner_bbox_3d)

        # Make sure it also accept more value after last index.
        padded_corner_bbox = tf.pad(
            self.corner_bbox, [[0, 0], [0, 2]]
        )  # Right pad 2 more value
        padded_xywh_bbox = tf.pad(self.xywh_bbox, [[0, 0], [0, 2]])
        self.assertAllClose(
            bbox.xywh_to_corners(padded_xywh_bbox), padded_corner_bbox
        )

        # Same for higher rank
        padded_corner_bbox_3d = tf.expand_dims(padded_corner_bbox, 0)
        padded_xywh_bbox_3d = tf.expand_dims(padded_xywh_bbox, 0)
        self.assertAllClose(
            bbox.xywh_to_corners(padded_xywh_bbox_3d), padded_corner_bbox_3d
        )

    def test_bbox_padding(self):
        bboxes = [[1, 2, 3, 4], [5, 6, 7, 8]]
        target_shape = [3, 4]
        result = bbox.pad_bbox_batch_to_shape(bboxes, target_shape)
        self.assertAllClose(result, [[1, 2, 3, 4], [5, 6, 7, 8], [-1, -1, -1, -1]])

        target_shape = [2, 5]
        result = bbox.pad_bbox_batch_to_shape(bboxes, target_shape)
        self.assertAllClose(result, [[1, 2, 3, 4, -1], [5, 6, 7, 8, -1]])

        # Make sure to raise error if the rank is different between bbox and target
        # shape
        with self.assertRaisesRegexp(
            ValueError, "Target shape should have same rank"
        ):
            bbox.pad_bbox_batch_to_shape(bboxes, [1, 2, 3])

        # Make sure raise error if the target shape is smaller
        target_shape = [3, 2]
        with self.assertRaisesRegexp(
            ValueError, "Target shape should be larger than bounding box shape"
        ):
            bbox.pad_bbox_batch_to_shape(bboxes, target_shape)
