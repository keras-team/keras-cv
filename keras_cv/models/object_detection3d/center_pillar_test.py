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

from keras_cv.layers.object_detection3d.voxelization import DynamicVoxelization
from keras_cv.models.object_detection3d.center_pillar import MultiHeadCenterPillar


class CenterPillarsTest(tf.test.TestCase):
    def get_point_net(self):
        return tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10),
                tf.keras.layers.Dense(20),
            ]
        )

    def test_center_pillar_call(self):
        voxel_net = DynamicVoxelization(
            point_net=self.get_point_net(),
            voxel_size=[0.1, 0.1, 1000],
            spatial_size=[-20, 20, -20, 20, -20, 20],
        )
        model = MultiHeadCenterPillar(
            backbone=None,
            voxel_net=voxel_net,
            multiclass_head=None,
            label_encoder=None,
            prediction_decoder=None,
        )
        point_xyz = tf.random.normal([32, 1000, 3])
        point_feature = tf.random.normal([32, 1000, 4])
        point_mask = tf.constant(True, shape=[3, 1000])
        outputs = model(point_xyz, point_feature, point_mask)
        for k, v in outputs.items():
            print(k, v.shape)
