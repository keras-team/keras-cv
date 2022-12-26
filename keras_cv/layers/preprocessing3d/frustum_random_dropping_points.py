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

from keras_cv.layers.preprocessing3d import base_augmentation_layer_3d
from keras_cv.ops import point_cloud

POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES
POINTCLOUD_LABEL_INDEX = base_augmentation_layer_3d.POINTCLOUD_LABEL_INDEX


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class FrustumRandomDroppingPoints(base_augmentation_layer_3d.BaseAugmentationLayer3D):
    """A preprocessing layer which randomly drops point within a randomly generated frustum during training.

    This layer will randomly select a point from the point cloud as the center of a frustum then generate a frustum based
    on r_distance, theta_width, and phi_width. Points inside the selected frustum are randomly dropped (setting all features to zero)
    based on drop_rate.
    The point_clouds tensor shape must be specific and cannot be dynamic.
    During inference time, the output will be identical to input. Call the layer with `training=True` to drop the input points.

    Input shape:
      point_clouds: 3D (multi frames) float32 Tensor with shape
        [num of frames, num of points, num of point features].
        The first 5 features are [x, y, z, class, range].
      bounding_boxes: 3D (multi frames) float32 Tensor with shape
        [num of frames, num of boxes, num of box features].
        The first 7 features are [x, y, z, dx, dy, dz, phi].

    Output shape:
      A dictionary of Tensors with the same shape as input Tensors.

    Arguments:
      r_distance: A float scalar sets the starting distance of a frustum.
      theta_width: A float scalar sets the theta width of a frustum.
      phi_width: A float scalar sets the phi width of a frustum.
      drop_rate: A float scalar sets the probability threshold for dropping the points.
    """

    def __init__(self, r_distance, theta_width, phi_width, drop_rate=None, **kwargs):
        super().__init__(**kwargs)

        if r_distance < 0:
            raise ValueError("r_distance must be >=0.")
        if theta_width < 0:
            raise ValueError("theta_width must be >=0.")
        if phi_width < 0:
            raise ValueError("phi_width must be >=0.")
        drop_rate = drop_rate if drop_rate else 0.0
        if drop_rate > 1:
            raise ValueError("drop_rate must be <=1.")

        self._r_distance = r_distance
        self._theta_width = theta_width
        self._phi_width = phi_width
        keep_probability = 1 - drop_rate
        self._keep_probability = keep_probability

    def get_config(self):
        return {
            "r_distance": self._r_distance,
            "theta_width": self._theta_width,
            "phi_width": self._phi_width,
            "drop_rate": 1 - self._keep_probability,
        }

    def get_random_transformation(self, point_clouds, **kwargs):
        # Randomly select a point from the first frame as the center of the frustum.
        valid_points = point_clouds[0, :, POINTCLOUD_LABEL_INDEX] > 0
        num_valid_points = tf.math.reduce_sum(tf.cast(valid_points, tf.int32))
        randomly_select_point_index = tf.random.uniform(
            (), minval=0, maxval=num_valid_points, dtype=tf.int32
        )
        randomly_select_frustum_center = tf.boolean_mask(
            point_clouds[0], valid_points, axis=0
        )[randomly_select_point_index, :POINTCLOUD_LABEL_INDEX]
        num_frames, num_points, _ = point_clouds.get_shape().as_list()
        frustum_mask = []
        for f in range(num_frames):
            frustum_mask.append(
                point_cloud.within_a_frustum(
                    point_clouds[f],
                    randomly_select_frustum_center,
                    self._r_distance,
                    self._theta_width,
                    self._phi_width,
                )[tf.newaxis, :, tf.newaxis]
            )
        frustum_mask = tf.concat(frustum_mask, axis=0)
        random_point_mask = (
            self._random_generator.random_uniform(
                [num_frames, num_points, 1], minval=0.0, maxval=1
            )
            < self._keep_probability
        )
        # Do not drop points outside the frustum mask.
        random_point_mask = tf.where(~frustum_mask, True, random_point_mask)
        return {"point_mask": random_point_mask}

    def augment_point_clouds_bounding_boxes(
        self, point_clouds, bounding_boxes, transformation, **kwargs
    ):
        point_mask = transformation["point_mask"]
        point_clouds = tf.where(point_mask, point_clouds, 0.0)
        return (point_clouds, bounding_boxes)
