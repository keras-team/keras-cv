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

from keras_cv.backend import keras
from keras_cv.layers.object_detection_3d.voxelization import DynamicVoxelization
from keras_cv.tests.test_case import TestCase


class VoxelizationTest(TestCase):
    def test_voxelization_output_shape_no_z(self):
        layer = DynamicVoxelization(
            voxel_size=[0.1, 0.1, 1000],
            spatial_size=[-20, 20, -20, 20, -20, 20],
        )
        point_xyz = tf.random.uniform(
            shape=[1, 1000, 3], minval=-5, maxval=5, dtype=tf.float32
        )
        point_feature = tf.random.uniform(
            shape=[1, 1000, 4], minval=-10, maxval=10, dtype=tf.float32
        )
        point_mask = tf.cast(
            tf.random.uniform(
                shape=[1, 1000], minval=0, maxval=2, dtype=tf.int32
            ),
            tf.bool,
        )
        output = layer(point_xyz, point_feature, point_mask)
        # (20 - (-20)) / 0.1 = 400, (20 - (-20) ) / 1000 = 0.4
        # the last dimension is replaced with MLP dimension, z dimension is
        # skipped
        self.assertEqual(output.shape, (1, 400, 400, 128))

    def test_voxelization_output_shape_with_z(self):
        layer = DynamicVoxelization(
            voxel_size=[0.1, 0.1, 1],
            spatial_size=[-20, 20, -20, 20, -15, 15],
        )
        point_xyz = tf.random.uniform(
            shape=[1, 1000, 3], minval=-5, maxval=5, dtype=tf.float32
        )
        point_feature = tf.random.uniform(
            shape=[1, 1000, 4], minval=-10, maxval=10, dtype=tf.float32
        )
        point_mask = tf.cast(
            tf.random.uniform(
                shape=[1, 1000], minval=0, maxval=2, dtype=tf.int32
            ),
            tf.bool,
        )
        output = layer(point_xyz, point_feature, point_mask)
        # (20 - (-20)) / 0.1 = 400, (20 - (-20) ) / 1000 = 0.4
        # (15 - (-15)) / 1 = 30
        # the last dimension is replaced with MLP dimension, z dimension is
        # skipped
        self.assertEqual(output.shape, (1, 400, 400, 30, 128))

    def test_voxelization_numerical(self):
        layer = DynamicVoxelization(
            voxel_size=[1.0, 1.0, 10.0],
            spatial_size=[-5, 5, -5, 5, -2, 2],
        )
        # Make the point net a no-op to allow us to verify the voxelization.
        layer.point_net_dense = keras.layers.Identity()
        # TODO(ianstenbit): use Identity here once it supports masking
        layer.point_net_norm = keras.layers.Lambda(lambda x: x)
        layer.point_net_activation = keras.layers.Identity()
        point_xyz = tf.constant(
            [
                [
                    [-4.9, -4.9, 0.0],
                    [4.4, 4.4, 0.0],
                ]
            ]
        )
        point_feature = tf.constant(
            [
                [
                    [1.0],
                    [2.0],
                ]
            ]
        )

        point_mask = tf.constant([True], shape=[1, 2])

        output = layer(point_xyz, point_feature, point_mask)
        # [-4.9, -4.9, 0] will the mapped to the upper leftmost voxel,
        # the first element is point feature,
        # the second / third element is -4.9 - (-5) = 0.1
        self.assertAllClose(output[0][0][0], [1.0, 0.1, 0.1, 0])
        # [4.4, 4.4, 0] will the mapped to the lower rightmost voxel,
        # the first element is point feature
        # the second / third element is 4.4 - 4 = 0.4, because the
        # voxel range is [-5, 4] for 10 voxels.
        self.assertAllClose(output[0][-1][-1], [2.0, 0.4, 0.4, 0])
