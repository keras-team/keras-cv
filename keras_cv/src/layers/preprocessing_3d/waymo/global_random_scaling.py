# Copyright 2022 Waymo LLC.
#
# Licensed under the terms in https://github.com/keras-team/keras-cv/blob/master/keras_cv/layers/preprocessing_3d/waymo/LICENSE  # noqa: E501

import tensorflow as tf

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import random
from keras_cv.src.bounding_box_3d import CENTER_XYZ_DXDYDZ_PHI
from keras_cv.src.layers.preprocessing_3d import base_augmentation_layer_3d

POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES


@keras_cv_export("keras_cv.layers.GlobalRandomScaling")
class GlobalRandomScaling(base_augmentation_layer_3d.BaseAugmentationLayer3D):
    """A preprocessing layer which randomly scales point clouds and bounding
    boxes along X, Y, and Z axes during training.

    This layer will randomly scale the whole scene along the  X, Y, and Z axes
    based on a randomly sampled scaling factor between [min_scaling_factor,
    max_scaling_factor] following a uniform distribution.

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
      x_factor: A tuple of float scalars or a float scalar sets the minimum and
        maximum scaling factors for the X axis.
      y_factor: A tuple of float scalars or a float scalar sets the minimum and
        maximum scaling factors for the Y axis.
      z_factor: A tuple of float scalars or a float scalar sets the minimum and
        maximum scaling factors for the Z axis.
    """

    def __init__(
        self,
        x_factor=None,
        y_factor=None,
        z_factor=None,
        preserve_aspect_ratio=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if not x_factor:
            min_x_factor = 1.0
            max_x_factor = 1.0
        elif type(x_factor) is float:
            min_x_factor = x_factor
            max_x_factor = x_factor
        else:
            min_x_factor = x_factor[0]
            max_x_factor = x_factor[1]
        if not y_factor:
            min_y_factor = 1.0
            max_y_factor = 1.0
        elif type(y_factor) is float:
            min_y_factor = y_factor
            max_y_factor = y_factor
        else:
            min_y_factor = y_factor[0]
            max_y_factor = y_factor[1]
        if not z_factor:
            min_z_factor = 1.0
            max_z_factor = 1.0
        elif type(z_factor) is float:
            min_z_factor = z_factor
            max_z_factor = z_factor
        else:
            min_z_factor = z_factor[0]
            max_z_factor = z_factor[1]

        if (
            min_x_factor < 0
            or max_x_factor < 0
            or min_y_factor < 0
            or max_y_factor < 0
            or min_z_factor < 0
            or max_z_factor < 0
        ):
            raise ValueError("min_factor and max_factor must be >=0.")
        if (
            min_x_factor > max_x_factor
            or min_y_factor > max_y_factor
            or min_z_factor > max_z_factor
        ):
            raise ValueError("min_factor must be less than max_factor.")
        if preserve_aspect_ratio:
            if min_x_factor != min_y_factor or min_y_factor != min_z_factor:
                raise ValueError(
                    "min_factor must be the same when preserve_aspect_ratio is "
                    "true."
                )
            if max_x_factor != max_y_factor or max_y_factor != max_z_factor:
                raise ValueError(
                    "max_factor must be the same when preserve_aspect_ratio is "
                    "true."
                )

        self._min_x_factor = min_x_factor
        self._max_x_factor = max_x_factor
        self._min_y_factor = min_y_factor
        self._max_y_factor = max_y_factor
        self._min_z_factor = min_z_factor
        self._max_z_factor = max_z_factor
        self._preserve_aspect_ratio = preserve_aspect_ratio

    def get_config(self):
        return {
            "x_factor": (
                self._min_x_factor,
                self._max_x_factor,
            ),
            "y_factor": (
                self._min_y_factor,
                self._max_y_factor,
            ),
            "z_factor": (
                self._min_z_factor,
                self._max_z_factor,
            ),
            "preserve_aspect_ratio": self._preserve_aspect_ratio,
        }

    def get_random_transformation(self, **kwargs):
        random_scaling_x = random.uniform(
            (),
            minval=self._min_x_factor,
            maxval=self._max_x_factor,
            dtype=self.compute_dtype,
            seed=self._random_generator,
        )
        random_scaling_y = random.uniform(
            (),
            minval=self._min_y_factor,
            maxval=self._max_y_factor,
            dtype=self.compute_dtype,
            seed=self._random_generator,
        )
        random_scaling_z = random.uniform(
            (),
            minval=self._min_z_factor,
            maxval=self._max_z_factor,
            dtype=self.compute_dtype,
            seed=self._random_generator,
        )
        if not self._preserve_aspect_ratio:
            return {
                "scale": tf.stack(
                    [random_scaling_x, random_scaling_y, random_scaling_z]
                )
            }
        else:
            return {
                "scale": tf.stack(
                    [random_scaling_x, random_scaling_x, random_scaling_x]
                )
            }

    def augment_point_clouds_bounding_boxes(
        self, point_clouds, bounding_boxes, transformation, **kwargs
    ):
        scale = transformation["scale"][tf.newaxis, tf.newaxis, :]
        point_clouds_xyz = point_clouds[..., :3] * scale
        point_clouds = tf.concat(
            [point_clouds_xyz, point_clouds[..., 3:]], axis=-1
        )

        bounding_boxes_xyzdxdydz = bounding_boxes[
            ..., : CENTER_XYZ_DXDYDZ_PHI.DZ + 1
        ] * tf.concat([scale] * 2, axis=-1)
        bounding_boxes = tf.concat(
            [
                bounding_boxes_xyzdxdydz,
                bounding_boxes[..., CENTER_XYZ_DXDYDZ_PHI.PHI :],
            ],
            axis=-1,
        )

        return (point_clouds, bounding_boxes)
