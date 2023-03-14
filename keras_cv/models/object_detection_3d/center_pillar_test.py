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

from keras_cv.layers.object_detection_3d.voxelization import DynamicVoxelization
from keras_cv.models.__internal__.unet import Block
from keras_cv.models.__internal__.unet import UNet
from keras_cv.models.object_detection_3d.center_pillar import (
    MultiClassDetectionHead,
)
from keras_cv.models.object_detection_3d.center_pillar import (
    MultiClassHeatmapDecoder,
)
from keras_cv.models.object_detection_3d.center_pillar import (
    MultiHeadCenterPillar,
)

down_block_configs = [(128, 6), (256, 2), (512, 2)]
up_block_configs = [512, 256, 256]


class CenterPillarTest(tf.test.TestCase):
    def get_point_net(self):
        return tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10),
                tf.keras.layers.Dense(20),
            ]
        )

    def build_centerpillar_unet(self, input_shape):
        input = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(
            128,
            1,
            1,
            padding="same",
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            kernel_regularizer=tf.keras.regularizers.L2(l2=1e-4),
        )(input)
        x = tf.keras.layers.BatchNormalization(
            beta_regularizer=tf.keras.regularizers.L2(l2=1e-8),
            gamma_regularizer=tf.keras.regularizers.L2(l2=1e-8),
        )(x)
        x = tf.keras.layers.ReLU()(x)
        x = Block(128, downsample=False, sync_bn=False)(x)
        output = UNet(
            x.shape[1:], down_block_configs, up_block_configs, sync_bn=False
        )(x)
        return tf.keras.Model(input, output)

    def test_center_pillar_call(self):
        voxel_net = DynamicVoxelization(
            point_net=self.get_point_net(),
            voxel_size=[0.1, 0.1, 1000],
            spatial_size=[-20, 20, -20, 20, -20, 20],
        )
        # dimensions computed from voxel_net
        unet = self.build_centerpillar_unet([400, 400, 20])
        decoder = MultiClassHeatmapDecoder(
            num_class=2,
            num_head_bin=[2, 2],
            anchor_size=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            max_pool_size=[3, 3],
            max_num_box=[3, 4],
            heatmap_threshold=[0.2, 0.2],
            voxel_size=voxel_net._voxel_size,
            spatial_size=voxel_net._spatial_size,
        )
        multiclass_head = MultiClassDetectionHead(
            num_class=2,
            num_head_bin=[2, 2],
        )
        model = MultiHeadCenterPillar(
            backbone=unet,
            voxel_net=voxel_net,
            multiclass_head=multiclass_head,
            label_encoder=None,
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
        self.assertEqual(outputs["class_1"].shape, [2, 400, 400, 12])
        self.assertEqual(outputs["class_2"].shape, [2, 400, 400, 12])

    def test_center_pillar_predict(self):
        voxel_net = DynamicVoxelization(
            point_net=self.get_point_net(),
            voxel_size=[0.1, 0.1, 1000],
            spatial_size=[-20, 20, -20, 20, -20, 20],
        )
        # dimensions computed from voxel_net
        unet = self.build_centerpillar_unet([400, 400, 20])
        decoder = MultiClassHeatmapDecoder(
            num_class=2,
            num_head_bin=[2, 2],
            anchor_size=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            max_pool_size=[3, 3],
            max_num_box=[3, 4],
            heatmap_threshold=[0.2, 0.2],
            voxel_size=voxel_net._voxel_size,
            spatial_size=voxel_net._spatial_size,
        )
        multiclass_head = MultiClassDetectionHead(
            num_class=2,
            num_head_bin=[2, 2],
        )
        model = MultiHeadCenterPillar(
            backbone=unet,
            voxel_net=voxel_net,
            multiclass_head=multiclass_head,
            label_encoder=None,
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
            training=False,
        )
        # max number boxes is 3
        self.assertEqual(outputs["3d_boxes"]["boxes"].shape, [2, 7, 7])
        self.assertEqual(outputs["3d_boxes"]["classes"].shape, [2, 7])
        self.assertEqual(outputs["3d_boxes"]["confidence"].shape, [2, 7])
        self.assertAllEqual(
            outputs["3d_boxes"]["classes"],
            tf.constant([1, 1, 1, 2, 2, 2, 2] * 2, shape=(2, 7)),
        )
