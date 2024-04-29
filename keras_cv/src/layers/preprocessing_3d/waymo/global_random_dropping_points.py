# Copyright 2022 Waymo LLC.
#
# Licensed under the terms in https://github.com/keras-team/keras-cv/blob/master/keras_cv/layers/preprocessing_3d/waymo/LICENSE  # noqa: E501

import tensorflow as tf

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import random
from keras_cv.src.layers.preprocessing_3d import base_augmentation_layer_3d

POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES


@keras_cv_export("keras_cv.layers.GlobalRandomDroppingPoints")
class GlobalRandomDroppingPoints(
    base_augmentation_layer_3d.BaseAugmentationLayer3D
):
    """A preprocessing layer which randomly drops point during training.

    This layer will randomly drop points based on keep_probability.

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
      drop_rate: A float scalar sets the probability threshold for dropping the
        points.
      exclude_classes: An optional int scalar or a list of ints. Points with the
        specified class(es) will not be dropped.

    """

    def __init__(self, drop_rate=None, exclude_classes=None, **kwargs):
        super().__init__(**kwargs)
        drop_rate = drop_rate if drop_rate else 0.0

        if not isinstance(exclude_classes, (tuple, list)):
            exclude_classes = [exclude_classes]

        if drop_rate > 1:
            raise ValueError("drop_rate must be <=1.")
        keep_probability = 1 - drop_rate
        self._keep_probability = keep_probability
        self._exclude_classes = exclude_classes

    def get_config(self):
        return {
            "drop_rate": 1 - self._keep_probability,
            "exclude_classes": self._exclude_classes,
        }

    def get_random_transformation(self, point_clouds, **kwargs):
        num_points = point_clouds.get_shape().as_list()[-2]
        # Generate mask along point dimension.
        random_point_mask = (
            random.uniform([1, num_points, 1], minval=0.0, maxval=1)
            < self._keep_probability
        )

        return {"point_mask": random_point_mask}

    def augment_point_clouds_bounding_boxes(
        self, point_clouds, bounding_boxes, transformation, **kwargs
    ):
        point_mask = transformation["point_mask"]

        # Do not add noise to points that are protected by setting the
        # corresponding point_noise = 1.0.
        protected_points = tf.zeros_like(point_clouds[0, :, -1], dtype=tf.bool)
        for excluded_class in self._exclude_classes:
            protected_points |= point_clouds[0, :, -1] == excluded_class

        point_mask = tf.where(
            protected_points[tf.newaxis, :, tf.newaxis], True, point_mask
        )
        point_clouds = tf.where(point_mask, point_clouds, 0.0)
        return (point_clouds, bounding_boxes)
