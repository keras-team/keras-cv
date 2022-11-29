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

import tensorflow as tf

from keras_cv.datasets.waymo import load
from keras_cv.datasets.waymo import transformer


class WaymoOpenDatasetTransformerTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.test_data_path = os.path.abspath(
            os.path.join(os.path.abspath(__file__), os.path.pardir, "test_data")
        )
        self.output_signature = transformer.WOD_FRAME_OUTPUT_SIGNATURE

    def test_load_and_transform(self):
        dataset = load.load(
            self.test_data_path,
            transformer.build_tensors_from_wod_frame,
            self.output_signature,
        )
        batched_dataset = dataset.batch(1)

        # Extract records into a list.
        dataset = list(dataset)
        self.assertEqual(len(dataset), 1)
        lidar_tensors = dataset[0]
        num_boxes = lidar_tensors["label_box"].shape[0]
        self.assertEqual(num_boxes, 16)
        self.assertEqual(lidar_tensors["frame_id"], 1030973309163114042)
        self.assertEqual(lidar_tensors["timestamp_micros"], 1550083467346370)
        self.assertEqual(lidar_tensors["timestamp_offset"], 0)
        self.assertGreater(lidar_tensors["timestamp_micros"], 0)
        self.assertAllEqual(
            lidar_tensors["label_box_detection_difficulty"],
            tf.zeros(num_boxes, dtype=tf.int32),
        )

        # Laser points.
        point_smooth_xyz_mean = tf.reduce_mean(
            lidar_tensors["point_smooth_xyz"], axis=0
        )
        self.assertAllClose(
            point_smooth_xyz_mean, lidar_tensors["pose"][:3, 3], atol=100
        )
        point_feature_mean = tf.reduce_mean(lidar_tensors["point_feature"], axis=0)
        self.assertAllGreater(point_feature_mean[0], 0)
        self.assertAllGreater(tf.abs(point_feature_mean[1]), 1e-6)
        self.assertAllGreater(point_feature_mean[2:4], 0)
        self.assertTrue(tf.math.reduce_all(lidar_tensors["point_mask"]))

        # Laser labels.
        self.assertEqual(lidar_tensors["label_box_id"].shape[0], num_boxes)
        self.assertEqual(lidar_tensors["label_box_meta"].shape[0], num_boxes)
        self.assertEqual(lidar_tensors["label_box_class"].shape[0], num_boxes)
        self.assertEqual(lidar_tensors["label_box_density"].shape[0], num_boxes)
        self.assertTrue(tf.math.reduce_all(lidar_tensors["label_box_mask"]))
        self.assertAllGreater(tf.math.reduce_max(lidar_tensors["label_point_class"]), 0)

        # Multi-frame tensors for augmentation.
        pointcloud, boxes = transformer.build_tensors_for_augmentation(batched_dataset)
        self.assertEqual(pointcloud.shape, [1, 1, 183142, 8])
        self.assertEqual(boxes.shape, [1, 1, 16, 11])
