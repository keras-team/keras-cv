# Copyright 2022 The KerasCV Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
from datetime import datetime

import numpy as np
import pytest
import tensorflow as tf

import keras_cv

num_points = 200000
num_boxes = 1000
box_dimension = 20.0


def get_points_boxes():
    points = tf.random.uniform(
        shape=[num_points, 2], minval=0, maxval=box_dimension, dtype=tf.float32
    )
    points_z = 5.0 * tf.ones(shape=[num_points, 1], dtype=tf.float32)
    points = tf.concat([points, points_z], axis=-1)
    boxes_x = tf.random.uniform(
        shape=[num_boxes, 1], minval=0, maxval=box_dimension - 1.0, dtype=tf.float32
    )
    boxes_y = tf.random.uniform(
        shape=[num_boxes, 1], minval=0, maxval=box_dimension - 1.0, dtype=tf.float32
    )
    boxes_dx = tf.random.uniform(
        shape=[num_boxes, 1], minval=0, maxval=5.0, dtype=tf.float32
    )
    boxes_dx = tf.math.minimum(10 - boxes_x, boxes_dx)
    boxes_dy = tf.random.uniform(
        shape=[num_boxes, 1], minval=0, maxval=5.0, dtype=tf.float32
    )
    boxes_dy = tf.math.minimum(10 - boxes_y, boxes_dy)
    boxes_z = 5.0 * tf.ones([num_boxes, 1], dtype=tf.float32)
    boxes_dz = 3.0 * tf.ones([num_boxes, 1], dtype=tf.float32)
    boxes_angle = tf.zeros([num_boxes, 1], dtype=tf.float32)
    boxes = tf.concat(
        [boxes_x, boxes_y, boxes_z, boxes_dx, boxes_dy, boxes_dz, boxes_angle], axis=-1
    )
    return points, boxes


class WithinBox3DTest(tf.test.TestCase):
    @pytest.mark.skipif(
        "TEST_CUSTOM_OPS" not in os.environ or os.environ["TEST_CUSTOM_OPS"] != "true",
        reason="Requires binaries compiled from source",
    )
    def test_unbatched_unrotated(self):
        boxes = np.array(
            [
                [0, 0, 0, 4, 4, 4, 0],
                [5, 5, 5, 1, 1, 1, 0],
            ]
        ).astype("float32")
        points = np.array(
            [
                [0, 0, 0],
                [0, 0, 2],
                # this point has z value larger than box top z
                [0, 0, 2.1],
                [2, 0, 0],
                [2.01, 0, 0],
                # this point belongs to 2nd box
                [5.5, 5.5, 5.5],
                # this point doesn't belong to 2nd box
                [5.6, 5.5, 5.5],
            ]
        ).astype("float32")
        res = keras_cv.ops.within_box3d_index(points, boxes)
        self.assertAllEqual([0, 0, -1, 0, -1, 1, -1], res)

    @pytest.mark.skipif(
        "TEST_CUSTOM_OPS" not in os.environ or os.environ["TEST_CUSTOM_OPS"] != "true",
        reason="Requires binaries compiled from source",
    )
    def test_unbatched_rotated(self):
        # a box rotated with 45 degree, the intersection with x and y axis
        # is [2*sqrt(2), 0] and [0, 2*sqrt(2)]
        boxes = np.array(
            [
                [0, 0, 0, 4, 4, 4, np.pi / 4],
            ]
        ).astype("float32")
        points = np.array(
            [
                [0, 0, 0],
                [0, 0, 2],
                # this point has z value larger than box top z
                [0, 0, 2.1],
                [2.82, 0, 0],
                # this point has x value larger than rotated box
                [2.83, 0, 0],
            ]
        ).astype("float32")
        res = keras_cv.ops.within_box3d_index(points, boxes)
        self.assertAllClose([0, 0, -1, 0, -1], res)

    @pytest.mark.skipif(
        "TEST_CUSTOM_OPS" not in os.environ or os.environ["TEST_CUSTOM_OPS"] != "true",
        reason="Requires binaries compiled from source",
    )
    def test_batched_unrotated(self):
        boxes = np.array(
            [
                [[0, 0, 0, 4, 4, 4, 0]],
                [[5, 5, 5, 1, 1, 1, 0]],
            ]
        ).astype("float32")
        points = np.array(
            [
                [
                    [0, 0, 0],
                    [0, 0, 2],
                    # this point has z value larger than box top z
                    [0, 0, 2.1],
                    [2, 0, 0],
                    [2.01, 0, 0],
                    # this point belongs to 2nd box
                    [5.5, 5.5, 5.5],
                    # this point doesn't belong to 2nd box
                    [5.6, 5.5, 5.5],
                ]
            ]
            * 2
        ).astype("float32")
        print(points.shape)
        res = keras_cv.ops.within_box3d_index(points, boxes)
        self.assertAllEqual(
            [[0, 0, -1, 0, -1, -1, -1], [-1, -1, -1, -1, -1, 0, -1]], res
        )

    @pytest.mark.skipif(
        "TEST_CUSTOM_OPS" not in os.environ or os.environ["TEST_CUSTOM_OPS"] != "true",
        reason="Requires binaries compiled from source",
    )
    def test_batched_rotated(self):
        # a box rotated with 45 degree, the intersection with x and y axis
        # is [2*sqrt(2), 0] and [0, 2*sqrt(2)]
        boxes = np.array(
            [
                [[0, 0, 0, 4, 4, 4, np.pi / 4]],
                [[5, 5, 5, 1, 1, 1, 0]],
            ]
        ).astype("float32")
        points = np.array(
            [
                [
                    [0, 0, 0],
                    [0, 0, 2],
                    # this point has z value larger than box top z
                    [0, 0, 2.1],
                    [2.82, 0, 0],
                    # this point has x value larger than rotated box
                    [2.83, 0, 0],
                ]
            ]
            * 2
        ).astype("float32")
        res = keras_cv.ops.within_box3d_index(points, boxes)
        self.assertAllEqual([[0, 0, -1, 0, -1], [-1, -1, -1, -1, -1]], res)

    @pytest.mark.skipif(
        "TEST_CUSTOM_OPS" not in os.environ or os.environ["TEST_CUSTOM_OPS"] != "true",
        reason="Requires binaries compiled from source",
    )
    def test_many_points(self):
        points, boxes = get_points_boxes()

        for _ in range(5):
            print(datetime.now())
            res = keras_cv.ops.within_box3d_index(points, boxes)
            self.assertAllClose(res.shape, points.shape[:1])

    @pytest.mark.skipif(
        "TEST_CUSTOM_OPS" not in os.environ or os.environ["TEST_CUSTOM_OPS"] != "true",
        reason="Requires binaries compiled from source",
    )
    def test_equal(self):
        for _ in range(10000):
            with tf.device("cpu:0"):
                box_center = tf.random.uniform(shape=[1, 3], minval=-10.0, maxval=10.0)
                box_dim = tf.random.uniform(shape=[1, 3], minval=0.1, maxval=10.0)
                boxes = tf.concat([box_center, box_dim, [[0.0]]], axis=-1)
                points = tf.random.normal([32, 3])
                res = keras_cv.ops.is_within_any_box3d(points, boxes)
                res_v2 = keras_cv.ops.is_within_any_box3d_v2(points, boxes)
                self.assertAllEqual(res, res_v2)
