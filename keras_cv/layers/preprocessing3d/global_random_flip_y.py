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

from keras_cv.bounding_box_3d import CENTER_XYZ_DXDYDZ_PHI
from keras_cv.layers.preprocessing3d import base_augmentation_layer_3d
from keras_cv.ops.point_cloud import wrap_angle_radians

POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class GlobalRandomFlipY(base_augmentation_layer_3d.BaseAugmentationLayer3D):
    """A preprocessing layer which flips point clouds and bounding boxes with respect to the X axis during training.

    This layer will flip the whole scene with respect to the X axis.
    During inference time, the output will be identical to input. Call the layer with `training=True` to flip the input.

    Input shape:
      point_clouds: 3D (multi frames) float32 Tensor with shape
        [num of frames, num of points, num of point features].
        The first 5 features are [x, y, z, class, range].
      bounding_boxes: 3D (multi frames) float32 Tensor with shape
        [num of frames, num of boxes, num of box features]. Boxes are expected
        to follow the CENTER_XYZ_DXDYDZ_PHI format. Refer to
        https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box_3d/formats.py
        for more details on supported bounding box formats.

    Output shape:
      A dictionary of Tensors with the same shape as input Tensors.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_config(self):
        return {}

    def augment_point_clouds_bounding_boxes(
        self, point_clouds, bounding_boxes, transformation, **kwargs
    ):
        point_clouds_y = -point_clouds[..., 1:2]
        point_clouds = tf.concat(
            [point_clouds[..., 0:1], point_clouds_y, point_clouds[..., 2:]], axis=-1
        )
        # Flip boxes.
        bounding_boxes_y = -bounding_boxes[
            ..., CENTER_XYZ_DXDYDZ_PHI.Y : CENTER_XYZ_DXDYDZ_PHI.Y + 1
        ]
        bounding_boxes_xyz = tf.concat(
            [
                bounding_boxes[
                    ..., CENTER_XYZ_DXDYDZ_PHI.X : CENTER_XYZ_DXDYDZ_PHI.X + 1
                ],
                bounding_boxes_y,
                bounding_boxes[
                    ..., CENTER_XYZ_DXDYDZ_PHI.Z : CENTER_XYZ_DXDYDZ_PHI.Z + 1
                ],
            ],
            axis=-1,
        )

        # Compensate rotation.
        bounding_boxes_heading = wrap_angle_radians(
            -bounding_boxes[
                ..., CENTER_XYZ_DXDYDZ_PHI.PHI : CENTER_XYZ_DXDYDZ_PHI.PHI + 1
            ]
        )
        bounding_boxes = tf.concat(
            [
                bounding_boxes_xyz,
                bounding_boxes[
                    ..., CENTER_XYZ_DXDYDZ_PHI.DX : CENTER_XYZ_DXDYDZ_PHI.DZ + 1
                ],
                bounding_boxes_heading,
                bounding_boxes[..., CENTER_XYZ_DXDYDZ_PHI.CLASS :],
            ],
            axis=-1,
        )

        return (point_clouds, bounding_boxes)
