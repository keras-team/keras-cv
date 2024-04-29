# Copyright 2022 Waymo LLC.
#
# Licensed under the terms in https://github.com/keras-team/keras-cv/blob/master/keras_cv/layers/preprocessing_3d/waymo/LICENSE  # noqa: E501

import tensorflow as tf

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import random
from keras_cv.src.bounding_box_3d import CENTER_XYZ_DXDYDZ_PHI
from keras_cv.src.layers.preprocessing_3d import base_augmentation_layer_3d
from keras_cv.src.point_cloud import coordinate_transform

POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES


@keras_cv_export("keras_cv.layers.GlobalRandomTranslation")
class GlobalRandomTranslation(
    base_augmentation_layer_3d.BaseAugmentationLayer3D
):
    """A preprocessing layer which randomly translates point clouds and bounding
    boxes along X, Y, and Z axes during training.

    This layer will randomly translate the whole scene along the X, Y,and Z axes
    based on three randomly sampled translation factors following three normal
    distributions centered at 0 with standard deviation [x_stddev, y_stddev,
    z_stddev].

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
      x_stddev: A float scalar sets the translation noise standard deviation
        along the X axis.
      y_stddev: A float scalar sets the translation noise standard deviation
        along the Y axis.
      z_stddev: A float scalar sets the translation noise standard deviation
        along the Z axis.
    """

    def __init__(self, x_stddev=None, y_stddev=None, z_stddev=None, **kwargs):
        super().__init__(**kwargs)
        x_stddev = x_stddev if x_stddev else 0.0
        y_stddev = y_stddev if y_stddev else 0.0
        z_stddev = z_stddev if z_stddev else 0.0
        if x_stddev < 0 or y_stddev < 0 or z_stddev < 0:
            raise ValueError("x_stddev, y_stddev, and z_stddev must be >=0.")

        self._x_stddev = x_stddev
        self._y_stddev = y_stddev
        self._z_stddev = z_stddev

    def get_config(self):
        return {
            "x_stddev": self._x_stddev,
            "y_stddev": self._x_stddev,
            "z_stddev": self._z_stddev,
        }

    def get_random_transformation(self, **kwargs):
        random_x_translation = random.normal(
            (),
            mean=0.0,
            stddev=self._x_stddev,
            dtype=self.compute_dtype,
            seed=self._random_generator,
        )
        random_y_translation = random.normal(
            (),
            mean=0.0,
            stddev=self._y_stddev,
            dtype=self.compute_dtype,
            seed=self._random_generator,
        )
        random_z_translation = random.normal(
            (),
            mean=0.0,
            stddev=self._z_stddev,
            dtype=self.compute_dtype,
            seed=self._random_generator,
        )
        return {
            "pose": tf.stack(
                [
                    random_x_translation,
                    random_y_translation,
                    random_z_translation,
                    0.0,
                    0.0,
                    0.0,
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
        bounding_boxes = tf.concat(
            [
                bounding_boxes_xyz,
                bounding_boxes[..., CENTER_XYZ_DXDYDZ_PHI.DX :],
            ],
            axis=-1,
        )

        return (point_clouds, bounding_boxes)
