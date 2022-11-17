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

POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES


class GlobalRandomScaling(base_augmentation_layer_3d.BaseAugmentationLayer3D):
    """A preprocessing layer which randomly scales point clouds and bounding boxes along
    X, Y, and Z axes during training.

    This layer will randomly scale the whole scene along the  X, Y, and Z axes based on a randomly sampled
    scaling factor between [min_scaling_factor, max_scaling_factor] following a uniform distribution.
    During inference time, the output will be identical to input. Call the layer with `training=True` to scale the input.

    Input shape:
      point_clouds: 3D (multi frames) float32 Tensor with shape
        [num of frames, num of points, num of point features].
        The first 5 features are [x, y, z, class, range].
      bounding_boxes:  3D (multi frames) float32 Tensor with shape
        [num of frames, num of boxes, num of box features].
        The first 7 features are [x, y, z, dx, dy, dz, phi].

    Output shape:
      A tuple of two Tensors (point_clouds, bounding_boxes) with the same shape as input Tensors.

    Arguments:
      min_scaling_factor: A float scaler or Tensor sets the minimum scaling factor.
      max_scaling_factor: A float scaler or Tensor sets the maximum scaling factor.
    """

    def __init__(self, min_scaling_factor, max_scaling_factor, **kwargs):
        super().__init__(**kwargs)
        if min_scaling_factor < 0 or max_scaling_factor < 0:
            raise ValueError("min_scaling_factor and max_scaling_factor must be >=0.")
        if min_scaling_factor > max_scaling_factor:
            raise ValueError("min_scaling_factor must be less than max_scaling_factor.")
        self._min_scaling_factor = min_scaling_factor
        self._max_scaling_factor = max_scaling_factor

    def get_random_transformation(self, **kwargs):
        random_scaling = self._random_generator.random_uniform(
            (), minval=self._min_scaling_factor, maxval=self._max_scaling_factor
        )
        return {"scale": random_scaling}

    def augment_point_clouds_bounding_boxes(
        self, point_clouds, bounding_boxes, transformation, **kwargs
    ):
        scale = transformation["scale"]
        point_clouds_xyz = point_clouds[..., :3] * scale
        point_clouds = tf.concat([point_clouds_xyz, point_clouds[..., 3:]], axis=-1)

        bounding_boxes_xyzdxdydz = bounding_boxes[..., :6] * scale
        bounding_boxes = tf.concat(
            [bounding_boxes_xyzdxdydz, bounding_boxes[..., 6:]], axis=-1
        )

        return (point_clouds, bounding_boxes)
