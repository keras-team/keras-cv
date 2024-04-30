# Copyright 2022 Waymo LLC.
#
# Licensed under the terms in https://github.com/keras-team/keras-cv/blob/master/keras_cv/layers/preprocessing_3d/waymo/LICENSE  # noqa: E501

import tensorflow as tf

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import random
from keras_cv.src.bounding_box_3d import CENTER_XYZ_DXDYDZ_PHI
from keras_cv.src.layers.preprocessing_3d import base_augmentation_layer_3d
from keras_cv.src.point_cloud import coordinate_transform
from keras_cv.src.point_cloud import wrap_angle_radians

POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES


@keras_cv_export("keras_cv.layers.GlobalRandomRotation")
class GlobalRandomRotation(base_augmentation_layer_3d.BaseAugmentationLayer3D):
    """A preprocessing layer which randomly rotates point clouds and bounding
    boxes along X, Y and Z axes during training.

    This layer will randomly rotate the whole scene along the X, Y and Z axes
    based on a randomly sampled rotation angle between [-max_rotation_angle,
    max_rotation_angle] (in radians) following a uniform distribution. During
    inference time, the output will be identical to input. Call the layer with
    `training=True` to rotate the input.

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

    Arguments:
      max_rotation_angle_x: A float scalar sets the maximum rotation angle (in
        radians) along X axis.
      max_rotation_angle_y: A float scalar sets the maximum rotation angle (in
        radians) along Y axis.
      max_rotation_angle_z: A float scalar sets the maximum rotation angle (in
        radians) along Z axis.

    """

    def __init__(
        self,
        max_rotation_angle_x=None,
        max_rotation_angle_y=None,
        max_rotation_angle_z=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        max_rotation_angle_x = (
            max_rotation_angle_x if max_rotation_angle_x else 0.0
        )
        max_rotation_angle_y = (
            max_rotation_angle_y if max_rotation_angle_y else 0.0
        )
        max_rotation_angle_z = (
            max_rotation_angle_z if max_rotation_angle_z else 0.0
        )

        if max_rotation_angle_x < 0:
            raise ValueError("max_rotation_angle_x must be >=0.")
        if max_rotation_angle_y < 0:
            raise ValueError("max_rotation_angle_y must be >=0.")
        if max_rotation_angle_z < 0:
            raise ValueError("max_rotation_angle_z must be >=0.")
        self._max_rotation_angle_x = max_rotation_angle_x
        self._max_rotation_angle_y = max_rotation_angle_y
        self._max_rotation_angle_z = max_rotation_angle_z

    def get_config(self):
        return {
            "max_rotation_angle_x": self._max_rotation_angle_x,
            "max_rotation_angle_y": self._max_rotation_angle_y,
            "max_rotation_angle_z": self._max_rotation_angle_z,
        }

    def get_random_transformation(self, **kwargs):
        random_rotation_x = random.uniform(
            (),
            minval=-self._max_rotation_angle_x,
            maxval=self._max_rotation_angle_x,
            dtype=self.compute_dtype,
            seed=self._random_generator,
        )
        random_rotation_y = random.uniform(
            (),
            minval=-self._max_rotation_angle_y,
            maxval=self._max_rotation_angle_y,
            dtype=self.compute_dtype,
            seed=self._random_generator,
        )
        random_rotation_z = random.uniform(
            (),
            minval=-self._max_rotation_angle_z,
            maxval=self._max_rotation_angle_z,
            dtype=self.compute_dtype,
            seed=self._random_generator,
        )
        return {
            "pose": tf.stack(
                [
                    0,
                    0,
                    0,
                    random_rotation_z,
                    random_rotation_x,
                    random_rotation_y,
                ],
                axis=0,
            )
        }

    def augment_point_clouds_bounding_boxes(
        self, point_clouds, bounding_boxes, transformation, **kwargs
    ):
        pose = transformation["pose"]
        point_clouds_xyz = coordinate_transform(point_clouds[..., :3], pose)
        point_clouds = tf.concat(
            [point_clouds_xyz, point_clouds[..., 3:]], axis=-1
        )

        bounding_boxes_xyz = coordinate_transform(
            bounding_boxes[..., : CENTER_XYZ_DXDYDZ_PHI.Z + 1], pose
        )
        bounding_boxes_heading = wrap_angle_radians(
            tf.expand_dims(
                bounding_boxes[..., CENTER_XYZ_DXDYDZ_PHI.PHI], axis=-1
            )
            - pose[3]
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
