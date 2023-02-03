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

import os

import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_cv import ops


class AngleTest(tf.test.TestCase):
    def test_wrap_angle_radians(self):
        self.assertAllClose(
            -np.pi + 0.1, ops.point_cloud.wrap_angle_radians(np.pi + 0.1)
        )
        self.assertAllClose(0.0, ops.point_cloud.wrap_angle_radians(2 * np.pi))


class Boxes3DTestCase(tf.test.TestCase, parameterized.TestCase):
    def test_convert_center_to_corners(self):
        boxes = tf.constant(
            [
                [[1, 2, 3, 4, 3, 6, 0], [1, 2, 3, 4, 3, 6, 0]],
                [[1, 2, 3, 4, 3, 6, np.pi / 2.0], [1, 2, 3, 4, 3, 6, np.pi / 2.0]],
            ]
        )
        corners = ops._center_xyzWHD_to_corner_xyz(boxes)
        self.assertEqual((2, 2, 8, 3), corners.shape)
        for i in [0, 1]:
            self.assertAllClose(-1, np.min(corners[0, i, :, 0]))
            self.assertAllClose(3, np.max(corners[0, i, :, 0]))
            self.assertAllClose(0.5, np.min(corners[0, i, :, 1]))
            self.assertAllClose(3.5, np.max(corners[0, i, :, 1]))
            self.assertAllClose(0, np.min(corners[0, i, :, 2]))
            self.assertAllClose(6, np.max(corners[0, i, :, 2]))

        for i in [0, 1]:
            self.assertAllClose(-0.5, np.min(corners[1, i, :, 0]))
            self.assertAllClose(2.5, np.max(corners[1, i, :, 0]))
            self.assertAllClose(0.0, np.min(corners[1, i, :, 1]))
            self.assertAllClose(4.0, np.max(corners[1, i, :, 1]))
            self.assertAllClose(0, np.min(corners[1, i, :, 2]))
            self.assertAllClose(6, np.max(corners[1, i, :, 2]))

    def test_within_box2d(self):
        boxes = tf.constant(
            [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=tf.float32
        )
        points = tf.constant(
            [
                [-0.5, -0.5],
                [0.5, -0.5],
                [1.5, -0.5],
                [1.5, 0.5],
                [1.5, 1.5],
                [0.5, 1.5],
                [-0.5, 1.5],
                [-0.5, 0.5],
                [1.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=tf.float32,
        )
        is_inside = ops.is_within_box2d(points, boxes)
        expected = [[False]] * 8 + [[True]] * 2
        self.assertAllEqual(expected, is_inside)

    def test_within_zero_box2d(self):
        bbox = tf.constant(
            [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], dtype=tf.float32
        )
        points = tf.constant(
            [
                [-0.5, -0.5],
                [0.5, -0.5],
                [1.5, -0.5],
                [1.5, 0.5],
                [1.5, 1.5],
                [0.5, 1.5],
                [-0.5, 1.5],
                [-0.5, 0.5],
                [1.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=tf.float32,
        )
        is_inside = ops.is_within_box2d(points, bbox)
        expected = [[False]] * 10
        self.assertAllEqual(expected, is_inside)

    def test_is_on_lefthand_side(self):
        v1 = tf.constant([[0.0, 0.0]], dtype=tf.float32)
        v2 = tf.constant([[1.0, 0.0]], dtype=tf.float32)
        p = tf.constant([[0.5, 0.5], [-1.0, -3], [-1.0, 1.0]], dtype=tf.float32)
        res = ops._is_on_lefthand_side(p, v1, v2)
        self.assertAllEqual([[True, False, True]], res)
        res = ops._is_on_lefthand_side(v1, v1, v2)
        self.assertAllEqual([[True]], res)
        res = ops._is_on_lefthand_side(v2, v1, v2)
        self.assertAllEqual([[True]], res)

    @parameterized.named_parameters(
        ("without_rotation", 0.0),
        ("with_rotation_1_rad", 1.0),
        ("with_rotation_2_rad", 2.0),
        ("with_rotation_3_rad", 3.0),
    )
    def test_box_area(self, angle):
        boxes = tf.constant(
            [
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]],
                [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
            ],
            dtype=tf.float32,
        )
        expected = [[1.0], [2.0], [4.0]]

        def _rotate(bbox, theta):
            rotation_matrix = tf.reshape(
                [tf.cos(theta), -tf.sin(theta), tf.sin(theta), tf.cos(theta)],
                shape=(2, 2),
            )
            return tf.matmul(bbox, rotation_matrix)

        rotated_bboxes = _rotate(boxes, angle)
        res = ops._box_area(rotated_bboxes)
        self.assertAllClose(expected, res)

    def test_within_box3d(self):
        num_points, num_boxes = 19, 4
        # rotate the first box by pi / 2 so dim_x and dim_y are swapped.
        # The last box is a cube rotated by 45 degrees.
        bboxes = tf.constant(
            [
                [1.0, 2.0, 3.0, 6.0, 0.4, 6.0, np.pi / 2],
                [4.0, 5.0, 6.0, 7.0, 0.8, 7.0, 0.0],
                [0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.0],
                [-10.0, -10.0, -10.0, 3.0, 3.0, 3.0, np.pi / 4],
            ],
            dtype=tf.float32,
        )
        points = tf.constant(
            [
                [1.0, 2.0, 3.0],  # box 0 (centroid)
                [0.8, 2.0, 3.0],  # box 0 (below x)
                [1.1, 2.0, 3.0],  # box 0 (above x)
                [1.3, 2.0, 3.0],  # box 0 (too far x)
                [0.7, 2.0, 3.0],  # box 0 (too far x)
                [4.0, 5.0, 6.0],  # box 1 (centroid)
                [4.0, 4.6, 6.0],  # box 1 (below y)
                [4.0, 5.4, 6.0],  # box 1 (above y)
                [4.0, 4.5, 6.0],  # box 1 (too far y)
                [4.0, 5.5, 6.0],  # box 1 (too far y)
                [0.4, 0.3, 0.2],  # box 2 (centroid)
                [0.4, 0.3, 0.1],  # box 2 (below z)
                [0.4, 0.3, 0.3],  # box 2 (above z)
                [0.4, 0.3, 0.0],  # box 2 (too far z)
                [0.4, 0.3, 0.4],  # box 2 (too far z)
                [5.0, 7.0, 8.0],  # none
                [1.0, 5.0, 3.6],  # box0, box1
                [-11.6, -10.0, -10.0],  # box3 (rotated corner point).
                [-11.4, -11.4, -10.0],  # not in box3, would be if not rotated.
            ],
            dtype=tf.float32,
        )
        expected_is_inside = np.array(
            [
                [True, False, False, False],
                [True, False, False, False],
                [True, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, True, False, False],
                [False, True, False, False],
                [False, True, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, True, False],
                [False, False, True, False],
                [False, False, True, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [True, True, False, False],
                [False, False, False, True],
                [False, False, False, False],
            ]
        )
        assert points.shape[0] == num_points
        assert bboxes.shape[0] == num_boxes
        assert expected_is_inside.shape[0] == num_points
        assert expected_is_inside.shape[1] == num_boxes
        is_inside = ops.is_within_box3d(points, bboxes)
        self.assertAllEqual([num_points, num_boxes], is_inside.shape)
        self.assertAllEqual(expected_is_inside, is_inside)
        # Add a batch dimension to the data and see that it still works
        # as expected.
        batch_size = 3
        points = tf.tile(points[tf.newaxis, ...], [batch_size, 1, 1])
        bboxes = tf.tile(bboxes[tf.newaxis, ...], [batch_size, 1, 1])
        is_inside = ops.is_within_box3d(points, bboxes)
        self.assertAllEqual([batch_size, num_points, num_boxes], is_inside.shape)
        for batch_idx in range(batch_size):
            self.assertAllEqual(expected_is_inside, is_inside[batch_idx])

    def testCoordinateTransform(self):
        # This is a validated test case from a real scene.
        #
        # A single point [1, 1, 3].
        point = tf.constant(
            [[[5736.94580078, 1264.85168457, 45.0271225]]], dtype=tf.float32
        )
        # Replicate the point to test broadcasting behavior.
        replicated_points = tf.tile(point, [2, 4, 1])

        # Pose of the car (x, y, z, yaw, roll, pitch).
        #
        # We negate the translations so that the coordinates are translated
        # such that the car is at the origin.
        pose = tf.constant(
            [
                -5728.77148438,
                -1264.42236328,
                -45.06399918,
                -3.10496902,
                0.03288471,
                0.00115049,
            ],
            dtype=tf.float32,
        )

        result = ops.coordinate_transform(replicated_points, pose)

        # We expect the point to be translated close to the car, and then rotated
        # mostly around the x-axis.
        # the result is device dependent, skip or ignore this test locally if it fails.
        expected = np.tile([[[-8.184512, -0.13086952, -0.04200769]]], [2, 4, 1])

        self.assertAllClose(expected, result)

    def testSphericalCoordinatesTransform(self):
        np_xyz = np.random.randn(5, 6, 3)
        points = tf.constant(np_xyz, dtype=tf.float32)
        spherical_coordinates = ops.spherical_coordinate_transform(points)

        # Convert coordinates back to xyz to verify.
        dist = spherical_coordinates[..., 0]
        theta = spherical_coordinates[..., 1]
        phi = spherical_coordinates[..., 2]

        x = dist * np.sin(theta) * np.cos(phi)
        y = dist * np.sin(theta) * np.sin(phi)
        z = dist * np.cos(theta)

        self.assertAllClose(x, np_xyz[..., 0])
        self.assertAllClose(y, np_xyz[..., 1])
        self.assertAllClose(z, np_xyz[..., 2])

    @pytest.mark.skipif(
        "TEST_CUSTOM_OPS" not in os.environ or os.environ["TEST_CUSTOM_OPS"] != "true",
        reason="Requires binaries compiled from source",
    )
    def test_group_points(self):
        # rotate the first box by pi / 2 so dim_x and dim_y are swapped.
        # The last box is a cube rotated by 45 degrees.
        with tf.device("cpu:0"):
            bboxes = tf.constant(
                [
                    [1.0, 2.0, 3.0, 6.0, 0.4, 6.0, np.pi / 2],
                    [4.0, 5.0, 6.0, 7.0, 0.8, 7.0, 0.0],
                    [0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.0],
                    [-10.0, -10.0, -10.0, 3.0, 3.0, 3.0, np.pi / 4],
                ],
                dtype=tf.float32,
            )
            points = tf.constant(
                [
                    [1.0, 2.0, 3.0],  # box 0 (centroid)
                    [0.8, 2.0, 3.0],  # box 0 (below x)
                    [1.1, 2.0, 3.0],  # box 0 (above x)
                    [1.3, 2.0, 3.0],  # box 0 (too far x)
                    [0.7, 2.0, 3.0],  # box 0 (too far x)
                    [4.0, 5.0, 6.0],  # box 1 (centroid)
                    [4.0, 4.61, 6.0],  # box 1 (below y)
                    [4.0, 5.39, 6.0],  # box 1 (above y)
                    [4.0, 4.5, 6.0],  # box 1 (too far y)
                    [4.0, 5.5, 6.0],  # box 1 (too far y)
                    [0.4, 0.3, 0.2],  # box 2 (centroid)
                    [0.4, 0.3, 0.1],  # box 2 (below z)
                    [0.4, 0.3, 0.29],  # box 2 (above z)
                    [0.4, 0.3, 0.0],  # box 2 (too far z)
                    [0.4, 0.3, 0.4],  # box 2 (too far z)
                    [5.0, 7.0, 8.0],  # none
                    [1.0, 5.0, 3.6],  # box0, box1
                    [-11.6, -10.0, -10.0],  # box3 (rotated corner point).
                    [-11.4, -11.4, -10.0],  # not in box3, would be if not rotated.
                ],
                dtype=tf.float32,
            )
            res = ops.group_points_by_boxes(points, bboxes)
            expected_result = tf.ragged.constant(
                [[0, 1, 2], [5, 6, 7, 16], [10, 11, 12], [17]]
            )
            self.assertAllClose(expected_result.flat_values, res.flat_values)

    def testWithinAFrustum(self):
        center = tf.constant([1.0, 1.0, 1.0])
        points = tf.constant([[0.0, 0.0, 0.0], [1.0, 2.0, 1.0], [1.0, 0.0, 1.0]])

        point_mask = ops.within_a_frustum(
            points, center, r_distance=1.0, theta_width=1.0, phi_width=1.0
        )
        target_point_mask = tf.constant([False, True, False])
        self.assertAllClose(point_mask, target_point_mask)

        point_mask = ops.within_a_frustum(
            points, center, r_distance=1.0, theta_width=3.14, phi_width=3.14
        )
        target_point_mask = tf.constant([False, True, True])
        self.assertAllClose(point_mask, target_point_mask)

        point_mask = ops.within_a_frustum(
            points, center, r_distance=3.0, theta_width=1.0, phi_width=1.0
        )
        target_point_mask = tf.constant([False, False, False])
        self.assertAllClose(point_mask, target_point_mask)
