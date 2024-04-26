# Copyright 2022 Waymo LLC.
#
# Licensed under the terms in https://github.com/keras-team/keras-cv/blob/master/keras_cv/layers/preprocessing_3d/waymo/LICENSE  # noqa: E501

import tensorflow as tf

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.bounding_box_3d import CENTER_XYZ_DXDYDZ_PHI
from keras_cv.src.layers.preprocessing_3d import base_augmentation_layer_3d
from keras_cv.src.point_cloud import is_within_any_box3d

POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES
ADDITIONAL_POINT_CLOUDS = base_augmentation_layer_3d.ADDITIONAL_POINT_CLOUDS
ADDITIONAL_BOUNDING_BOXES = base_augmentation_layer_3d.ADDITIONAL_BOUNDING_BOXES
POINTCLOUD_LABEL_INDEX = base_augmentation_layer_3d.POINTCLOUD_LABEL_INDEX


@keras_cv_export("keras_cv.layers.SwapBackground")
class SwapBackground(base_augmentation_layer_3d.BaseAugmentationLayer3D):
    """A preprocessing layer which swaps the backgrounds of two scenes during
    training.

    This layer will extract object point clouds and bounding boxes from an
    additional scene and paste it on to the training scene while removing the
    objects in the training scene. First, removing all the objects point clouds
    and bounding boxes in the training scene. Second, extracting object point
    clouds and bounding boxes from an additional scene. Third, removing
    backgrounds points clouds in the training scene that overlap with the
    additional object bounding boxes. Last, pasting the additional object point
    clouds and bounding boxes to the training background scene.

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
      A tuple of two Tensors (point_clouds, bounding_boxes) with the same shape
      as input Tensors.

    """

    def __init__(self, **kwargs):
        # TODO(ianstenbit): Support the model input format.
        super().__init__(**kwargs)
        self.auto_vectorize = False

    def get_config(self):
        return {}

    def get_random_transformation(
        self,
        point_clouds,
        bounding_boxes,
        additional_point_clouds,
        additional_bounding_boxes,
        **kwargs
    ):
        # Use the current frame bounding boxes to determine valid bounding
        # boxes.
        bounding_boxes = tf.boolean_mask(
            bounding_boxes,
            bounding_boxes[0, :, CENTER_XYZ_DXDYDZ_PHI.CLASS] > 0,
            axis=1,
        )
        additional_bounding_boxes = tf.boolean_mask(
            additional_bounding_boxes,
            additional_bounding_boxes[0, :, CENTER_XYZ_DXDYDZ_PHI.CLASS] > 0,
            axis=1,
        )

        # Remove objects in point_clouds.
        objects_points_in_point_clouds = is_within_any_box3d(
            point_clouds[..., :3],
            bounding_boxes[..., : CENTER_XYZ_DXDYDZ_PHI.CLASS],
            keepdims=True,
        )
        point_clouds = tf.where(
            ~objects_points_in_point_clouds, point_clouds, 0.0
        )

        # Extract objects from additional_point_clouds.
        objects_points_in_additional_point_clouds = is_within_any_box3d(
            additional_point_clouds[..., :3],
            additional_bounding_boxes[..., : CENTER_XYZ_DXDYDZ_PHI.CLASS],
            keepdims=True,
        )
        additional_point_clouds = tf.where(
            objects_points_in_additional_point_clouds,
            additional_point_clouds,
            0.0,
        )

        # Remove background points in point_clouds overlaps with
        # additional_bounding_boxes.
        points_overlaps_additional_bounding_boxes = is_within_any_box3d(
            point_clouds[..., :3],
            additional_bounding_boxes[..., : CENTER_XYZ_DXDYDZ_PHI.CLASS],
            keepdims=True,
        )
        point_clouds = tf.where(
            ~points_overlaps_additional_bounding_boxes, point_clouds, 0.0
        )
        return {
            POINT_CLOUDS: point_clouds,
            ADDITIONAL_POINT_CLOUDS: additional_point_clouds,
            ADDITIONAL_BOUNDING_BOXES: additional_bounding_boxes,
        }

    def augment_point_clouds_bounding_boxes(
        self, point_clouds, bounding_boxes, transformation, **kwargs
    ):
        original_bounding_boxes_shape = bounding_boxes.get_shape().as_list()
        original_point_clouds_shape = point_clouds.get_shape().as_list()
        point_clouds = transformation[POINT_CLOUDS]
        additional_point_clouds = transformation[ADDITIONAL_POINT_CLOUDS]
        num_frames = original_point_clouds_shape[0]
        point_clouds_list = []
        for frame_index in range(num_frames):
            background_point_clouds = tf.boolean_mask(
                point_clouds[frame_index],
                point_clouds[frame_index, :, POINTCLOUD_LABEL_INDEX] > 0,
                axis=0,
            )
            object_point_clouds = tf.boolean_mask(
                additional_point_clouds[frame_index],
                additional_point_clouds[frame_index, :, POINTCLOUD_LABEL_INDEX]
                > 0,
                axis=0,
            )

            point_clouds_list += [
                tf.concat(
                    [object_point_clouds, background_point_clouds], axis=0
                )
            ]

        point_clouds = tf.ragged.stack(point_clouds_list)
        bounding_boxes = tf.RaggedTensor.from_tensor(
            transformation[ADDITIONAL_BOUNDING_BOXES]
        )
        return (
            point_clouds.to_tensor(shape=original_point_clouds_shape),
            bounding_boxes.to_tensor(shape=original_bounding_boxes_shape),
        )

    def _augment(self, inputs):
        result = inputs
        point_clouds = inputs[POINT_CLOUDS]
        bounding_boxes = inputs[BOUNDING_BOXES]
        additional_point_clouds = inputs[ADDITIONAL_POINT_CLOUDS]
        additional_bounding_boxes = inputs[ADDITIONAL_BOUNDING_BOXES]

        transformation = self.get_random_transformation(
            point_clouds=point_clouds,
            bounding_boxes=bounding_boxes,
            additional_point_clouds=additional_point_clouds,
            additional_bounding_boxes=additional_bounding_boxes,
        )
        point_clouds, bounding_boxes = self.augment_point_clouds_bounding_boxes(
            point_clouds,
            bounding_boxes=bounding_boxes,
            transformation=transformation,
        )
        result.update(
            {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        )
        return result
