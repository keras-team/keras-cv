# Copyright 2022 Waymo LLC.
#
# Licensed under the terms in https://github.com/keras-team/keras-cv/blob/master/keras_cv/layers/preprocessing_3d/waymo/LICENSE  # noqa: E501

import tensorflow as tf

from keras_cv.src import point_cloud
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.layers.preprocessing_3d import base_augmentation_layer_3d

POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES
POINTCLOUD_LABEL_INDEX = base_augmentation_layer_3d.POINTCLOUD_LABEL_INDEX
POINTCLOUD_FEATURE_INDEX = base_augmentation_layer_3d.POINTCLOUD_FEATURE_INDEX


@keras_cv_export("keras_cv.layers.FrustumRandomPointFeatureNoise")
class FrustumRandomPointFeatureNoise(
    base_augmentation_layer_3d.BaseAugmentationLayer3D
):
    """A preprocessing layer which randomly add noise to point features within a
    randomly generated frustum during training.

    This layer will randomly select a point from the point cloud as the center
    of a frustum then generate a frustum based on r_distance, theta_width, and
    phi_width. Uniformly sampled features noise from [1-max_noise_level,
    1+max_noise_level] will be multiplied to points inside the selected frustum.
    Here, we perturb point features other than (x, y, z, class). The
    point_clouds tensor shape must be specific and cannot be dynamic. During
    inference time, the output will be identical to input. Call the layer with
    `training=True` to add noise to the input points.

    Input shape:
      point_clouds: 3D (multi frames) float32 Tensor with shape
        [num of frames, num of points, num of point features].
        The first 4 features are [x, y, z, class, additional features].
      bounding_boxes: 3D (multi frames) float32 Tensor with shape
        [num of frames, num of boxes, num of box features]. Boxes are expected
        to follow the CENTER_XYZ_DXDYDZ_PHI format. Refer to
        https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box_3d/formats.py
        for more details on supported bounding box formats.

    Output shape:
      A dictionary of Tensors with the same shape as input Tensors.

    Arguments:
      r_distance: A float scalar sets the starting distance of a frustum.
      theta_width: A float scalar sets the theta width of a frustum.
      phi_width: A float scalar sets the phi width of a frustum.
      max_noise_level: A float scalar sets the sampled feature noise range
        [1-max_noise_level, 1+max_noise_level].
      exclude_classes: An optional int scalar or a list of ints. Points with the
        specified class(es) will not be modified.

    """

    def __init__(
        self,
        r_distance,
        theta_width,
        phi_width,
        max_noise_level=None,
        exclude_classes=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if not isinstance(exclude_classes, (tuple, list)):
            exclude_classes = [exclude_classes]

        if r_distance < 0:
            raise ValueError("r_distance must be >=0.")
        if theta_width < 0:
            raise ValueError("theta_width must be >=0.")
        if phi_width < 0:
            raise ValueError("phi_width must be >=0.")
        max_noise_level = max_noise_level if max_noise_level else 0.0
        if max_noise_level < 0 or max_noise_level > 1:
            raise ValueError("max_noise_level must be >=0 and <=1.")

        self._r_distance = r_distance
        self._theta_width = theta_width
        self._phi_width = phi_width
        self._max_noise_level = max_noise_level
        self._exclude_classes = exclude_classes

    def get_config(self):
        return {
            "r_distance": self._r_distance,
            "theta_width": self._theta_width,
            "phi_width": self._phi_width,
            "max_noise_level": self._max_noise_level,
            "exclude_classes": self._exclude_classes,
        }

    def get_random_transformation(self, point_clouds, **kwargs):
        # Randomly select a point from the first frame as the center of the
        # frustum.
        valid_points = point_clouds[0, :, POINTCLOUD_LABEL_INDEX] > 0
        num_valid_points = tf.math.reduce_sum(tf.cast(valid_points, tf.int32))
        randomly_select_point_index = tf.random.uniform(
            (), minval=0, maxval=num_valid_points, dtype=tf.int32
        )
        randomly_select_frustum_center = tf.boolean_mask(
            point_clouds[0], valid_points, axis=0
        )[randomly_select_point_index, :POINTCLOUD_LABEL_INDEX]

        (
            num_frames,
            num_points,
            num_features,
        ) = point_clouds.get_shape().as_list()
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
        feature_noise = tf.random.uniform(
            [num_frames, num_points, num_features - POINTCLOUD_FEATURE_INDEX],
            minval=1 - self._max_noise_level,
            maxval=1 + self._max_noise_level,
        )
        noise = tf.concat(
            [
                tf.ones([num_frames, num_points, POINTCLOUD_FEATURE_INDEX]),
                feature_noise,
            ],
            axis=-1,
        )
        # Do add feature noise outside the frustum mask.
        random_point_noise = tf.where(~frustum_mask, 1.0, noise)
        random_point_noise = tf.cast(
            random_point_noise, dtype=self.compute_dtype
        )
        return {"point_noise": random_point_noise}

    def augment_point_clouds_bounding_boxes(
        self, point_clouds, bounding_boxes, transformation, **kwargs
    ):
        point_noise = transformation["point_noise"]

        # Do not add noise to points that are protected by setting the
        # corresponding point_noise = 1.0.
        protected_points = tf.zeros_like(point_clouds[..., -1], dtype=tf.bool)
        for excluded_class in self._exclude_classes:
            protected_points |= point_clouds[..., -1] == excluded_class

        no_noise = tf.ones_like(point_noise, point_noise.dtype)
        point_noise = tf.where(
            protected_points[:, :, tf.newaxis], no_noise, point_noise
        )
        point_clouds *= point_noise
        return (point_clouds, bounding_boxes)
