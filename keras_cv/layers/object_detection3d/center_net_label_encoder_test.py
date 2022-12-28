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

from keras_cv.layers.object_detection3d.center_net_label_encoder import (
    CenterNetLabelEncoder,
)


class CenterNetLabelEncoderTest(tf.test.TestCase):
    def test_voxelization_output_shape_no_z(self):
        layer = CenterNetLabelEncoder(
            voxel_size=[0.1, 0.1, 1000],
            min_radius=[0.8, 0.8, 0.0],
            max_radius=[8.0, 8.0, 0.0],
            spatial_size=[-20, 20, -20, 20, -20, 20],
            classes=2,
            top_k_heatmap=[10, 20],
        )
        box_3d = tf.random.uniform(
            shape=[2, 100, 7], minval=-5, maxval=5, dtype=tf.float32
        )
        box_classes = tf.random.uniform(
            shape=[2, 100], minval=0, maxval=2, dtype=tf.int32
        )
        box_mask = tf.constant(True, shape=[2, 100])
        output = layer(box_3d, box_classes, box_mask)
        # # (20 - (-20)) / 0.1 = 400
        self.assertEqual(output[0]["class_1"].shape, [2, 400, 400])
        self.assertEqual(output[0]["class_2"].shape, [2, 400, 400])
        self.assertEqual(output[1]["class_1"].shape, [2, 400, 400, 7])
        self.assertEqual(output[1]["class_2"].shape, [2, 400, 400, 7])
        # last dimension only has x, y
        self.assertEqual(output[2]["class_1"].shape, [2, 10, 2])
        self.assertEqual(output[2]["class_2"].shape, [2, 20, 2])

    def test_voxelization_output_shape_with_z(self):
        layer = CenterNetLabelEncoder(
            voxel_size=[0.1, 0.1, 10],
            min_radius=[0.8, 0.8, 0.0],
            max_radius=[8.0, 8.0, 0.0],
            spatial_size=[-20, 20, -20, 20, -20, 20],
            classes=2,
            top_k_heatmap=[10, 20],
        )
        box_3d = tf.random.uniform(
            shape=[2, 100, 7], minval=-5, maxval=5, dtype=tf.float32
        )
        box_classes = tf.random.uniform(
            shape=[2, 100], minval=0, maxval=2, dtype=tf.int32
        )
        box_mask = tf.constant(True, shape=[2, 100])
        output = layer(box_3d, box_classes, box_mask)
        # # (20 - (-20)) / 0.1 = 400
        self.assertEqual(output[0]["class_1"].shape, [2, 400, 400, 4])
        self.assertEqual(output[0]["class_2"].shape, [2, 400, 400, 4])
        self.assertEqual(output[1]["class_1"].shape, [2, 400, 400, 4, 7])
        self.assertEqual(output[1]["class_2"].shape, [2, 400, 400, 4, 7])
        # last dimension has x, y, z
        self.assertEqual(output[2]["class_1"].shape, [2, 10, 3])
        self.assertEqual(output[2]["class_2"].shape, [2, 20, 3])

    def test_voxelization_output_shape_missing_topk(self):
        layer = CenterNetLabelEncoder(
            voxel_size=[0.1, 0.1, 1000],
            min_radius=[0.8, 0.8, 0.0],
            max_radius=[8.0, 8.0, 0.0],
            spatial_size=[-20, 20, -20, 20, -20, 20],
            classes=2,
            top_k_heatmap=[10, 0],
        )
        box_3d = tf.random.uniform(
            shape=[2, 100, 7], minval=-5, maxval=5, dtype=tf.float32
        )
        box_classes = tf.random.uniform(
            shape=[2, 100], minval=0, maxval=2, dtype=tf.int32
        )
        box_mask = tf.constant(True, shape=[2, 100])
        output = layer(box_3d, box_classes, box_mask)
        # # (20 - (-20)) / 0.1 = 400
        self.assertEqual(output[0]["class_1"].shape, [2, 400, 400])
        self.assertEqual(output[0]["class_2"].shape, [2, 400, 400])
        self.assertEqual(output[1]["class_1"].shape, [2, 400, 400, 7])
        self.assertEqual(output[1]["class_2"].shape, [2, 400, 400, 7])
        # last dimension only has x, y
        self.assertEqual(output[2]["class_1"].shape, [2, 10, 2])
        self.assertEqual(output[2]["class_2"], None)
