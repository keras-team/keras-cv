# Copyright 2022 Waymo LLC.
#
# Licensed under the terms in https://github.com/keras-team/keras-cv/blob/master/keras_cv/layers/preprocessing_3d/waymo/LICENSE  # noqa: E501

import tensorflow as tf

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.bounding_box_3d import CENTER_XYZ_DXDYDZ_PHI
from keras_cv.src.layers.preprocessing_3d import base_augmentation_layer_3d
from keras_cv.src.point_cloud import wrap_angle_radians

POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES


@keras_cv_export("keras_cv.layers.GlobalRandomFlip")
class GlobalRandomFlip(base_augmentation_layer_3d.BaseAugmentationLayer3D):
    """A preprocessing layer which flips point clouds and bounding boxes with
    respect to the specified axis during training.

    This layer will flip the whole scene with respect to the specified axes.
    Note that this layer currently only supports flipping over the Y axis.

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

    Args:
      flip_x: whether to flip over the X axis, defaults to False.
      flip_y: whether to flip over the Y axis, defaults to True.
      flip_z: whether to flip over the Z axis, defaults to False.
    """

    def __init__(self, flip_x=False, flip_y=True, flip_z=False, **kwargs):
        if flip_x or flip_z:
            raise ValueError(
                "GlobalRandomFlip currently only supports flipping over the Y "
                f"axis. Received flip_x={flip_x}, flip_y={flip_y}, "
                f"flip_z={flip_z}."
            )

        if not (flip_x or flip_y or flip_z):
            raise ValueError("GlobalRandomFlip must flip over at least 1 axis.")
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.flip_z = flip_z

        super().__init__(**kwargs)

    def augment_point_clouds_bounding_boxes(
        self, point_clouds, bounding_boxes, transformation, **kwargs
    ):
        point_clouds_y = -point_clouds[..., 1:2]
        point_clouds = tf.concat(
            [point_clouds[..., 0:1], point_clouds_y, point_clouds[..., 2:]],
            axis=-1,
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

    def get_config(self):
        return {
            "flip_x": self.flip_x,
            "flip_y": self.flip_y,
            "flip_z": self.flip_z,
        }
