
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

from keras_cv.layers.preprocessing3d.base_augmentation_layer_3d import (
    BaseAugmentationLayer3D,
)
from keras_cv.ops.point_cloud import coordinate_transform, wrap_angle_rad


POINT_CLOUDS = "point_clouds"
BOUNDING_BOXES = "bounding_boxes"

class GlobalZRotation(BaseAugmentationLayer3D):
    """A preprocessing layer which randomly rotates point clouds and bounding boxes along
    Z axis during training.

    This layer will randomly rotate the whole scene along the Z axis based on a randomly sampled 
    rotation angle between [-max_rotation_angle, max_rotation_angle] following a uniform distribution.
    During inference time, the output will be identical to input. Call the layer with `training=True` to rotate the input.

    Input shape:
      point_clouds: 2D (single frame) or 3D (multi frames) float32 Tensor with shape 
        [..., num of points, num of point features]. 
        The first 5 features are [x, y, z, class, range].
      bounding_boxes: 2D (single frame) or 3D (multi frames) float32 Tensor with shape
        [..., num of boxes, num of box features]. 
        The first 7 features are [x, y, z, dx, dy, dz, phi].
    
    Output shape:
      A tuple of two Tensors (point_clouds, bounding_boxes) with the same shape as input Tensors.

    Arguments:
      max_rotation_angle: A float scaler or Tensor sets the maximum rotation angle. 
    """
    def __init__(self, max_rotation_angle, **kwargs):
        super().__init__(**kwargs)
        if max_rotation_angle<0:
            raise ValueError(
                "max_rotation_angle must be >=0."
            )
        self._max_rotation_angle = max_rotation_angle

    def get_random_transformation(self, **kwargs):
        random_rotation_z = self._random_generator.random_uniform(
            (), minval=-self._max_rotation_angle, maxval=self._max_rotation_angle
        )
        return {"pose": tf.stack([0, 0, 0, random_rotation_z, 0, 0], axis=0)}

    def augment_point_clouds_bounding_boxes(
        self, point_clouds, bounding_boxes, transformation, **kwargs
    ):
        pose = transformation["pose"]
        point_clouds_xyz = coordinate_transform(point_clouds[..., :3], pose)
        point_clouds = tf.concat([point_clouds_xyz, point_clouds[..., 3:]], axis=-1)
        
        bounding_boxes_xyz = coordinate_transform(bounding_boxes[..., :3], pose)
        bounding_boxes_heading = wrap_angle_rad(bounding_boxes[..., 6:7] - pose[3])
        bounding_boxes = tf.concat([
            bounding_boxes_xyz, bounding_boxes[..., 3:6], 
            bounding_boxes_heading, bounding_boxes[..., 7:]], axis=-1)

        return (point_clouds, bounding_boxes)
