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

import pytest
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config

from keras_cv.backend import keras
from keras_cv.backend.config import multi_backend
from keras_cv.layers.object_detection_3d.voxelization import DynamicVoxelization
from keras_cv.models.object_detection_3d.center_pillar import (
    MultiClassDetectionHead,
)
from keras_cv.models.object_detection_3d.center_pillar import (
    MultiClassHeatmapDecoder,
)
from keras_cv.models.object_detection_3d.center_pillar import (
    MultiHeadCenterPillar,
)
from keras_cv.models.object_detection_3d.center_pillar_backbone import (
    CenterPillarBackbone,
)
from keras_cv.tests.test_case import TestCase

np_config.enable_numpy_behavior()


@pytest.mark.skipif(
    multi_backend() and keras.backend.backend() == "torch",
    reason="CenterPillar does not yet support PyTorch.",
)
class CenterPillarTest(TestCase):
    def test_center_pillar_call(self):
        voxel_net = DynamicVoxelization(
            voxel_size=[0.1, 0.1, 1000],
            spatial_size=[-20, 20, -20, 20, -20, 20],
        )
        # dimensions computed from voxel_net
        backbone = CenterPillarBackbone.from_preset(
            "center_pillar_waymo_open_dataset"
        )
        decoder = MultiClassHeatmapDecoder(
            num_classes=2,
            num_head_bin=[2, 2],
            anchor_size=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            max_pool_size=[3, 3],
            max_num_box=[3, 4],
            heatmap_threshold=[0.2, 0.2],
            voxel_size=voxel_net._voxel_size,
            spatial_size=voxel_net._spatial_size,
        )
        multiclass_head = MultiClassDetectionHead(
            num_classes=2,
            num_head_bin=[2, 2],
        )
        model = MultiHeadCenterPillar(
            backbone=backbone,
            voxel_net=voxel_net,
            multiclass_head=multiclass_head,
            prediction_decoder=decoder,
        )
        point_xyz = tf.random.normal([2, 1000, 3])
        point_feature = tf.random.normal([2, 1000, 4])
        point_mask = tf.constant(True, shape=[2, 1000])
        outputs = model(
            {
                "point_xyz": point_xyz,
                "point_feature": point_feature,
                "point_mask": point_mask,
            },
            training=True,
        )
        self.assertEqual(outputs["class_1"].shape, (2, 400, 400, 12))
        self.assertEqual(outputs["class_2"].shape, (2, 400, 400, 12))

    def test_center_pillar_predict(self):
        voxel_net = DynamicVoxelization(
            voxel_size=[0.1, 0.1, 1000],
            spatial_size=[-20, 20, -20, 20, -20, 20],
        )
        backbone = CenterPillarBackbone.from_preset(
            "center_pillar_waymo_open_dataset"
        )
        decoder = MultiClassHeatmapDecoder(
            num_classes=2,
            num_head_bin=[2, 2],
            anchor_size=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            max_pool_size=[3, 3],
            max_num_box=[3, 4],
            heatmap_threshold=[0.2, 0.2],
            voxel_size=voxel_net._voxel_size,
            spatial_size=voxel_net._spatial_size,
        )
        multiclass_head = MultiClassDetectionHead(
            num_classes=2,
            num_head_bin=[2, 2],
        )
        model = MultiHeadCenterPillar(
            backbone=backbone,
            voxel_net=voxel_net,
            multiclass_head=multiclass_head,
            prediction_decoder=decoder,
        )
        point_xyz = tf.random.normal([2, 1000, 3])
        point_feature = tf.random.normal([2, 1000, 4])
        point_mask = tf.constant(True, shape=[2, 1000])
        outputs = model.predict(
            {
                "point_xyz": point_xyz,
                "point_feature": point_feature,
                "point_mask": point_mask,
            }
        )
        # max number boxes is 3
        self.assertEqual(outputs["3d_boxes"]["boxes"].shape, (2, 7, 7))
        self.assertEqual(outputs["3d_boxes"]["classes"].shape, (2, 7))
        self.assertEqual(outputs["3d_boxes"]["confidence"].shape, (2, 7))
        self.assertAllEqual(
            outputs["3d_boxes"]["classes"],
            tf.constant([1, 1, 1, 2, 2, 2, 2] * 2, shape=(2, 7)),
        )
