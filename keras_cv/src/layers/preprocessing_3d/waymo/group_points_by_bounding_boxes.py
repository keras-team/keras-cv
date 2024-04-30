# Copyright 2022 Waymo LLC.
#
# Licensed under the terms in https://github.com/keras-team/keras-cv/blob/master/keras_cv/layers/preprocessing_3d/waymo/LICENSE  # noqa: E501

import tensorflow as tf

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.bounding_box_3d import CENTER_XYZ_DXDYDZ_PHI
from keras_cv.src.layers.preprocessing_3d import base_augmentation_layer_3d
from keras_cv.src.point_cloud import group_points_by_boxes
from keras_cv.src.point_cloud import is_within_box3d

POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES
OBJECT_POINT_CLOUDS = base_augmentation_layer_3d.OBJECT_POINT_CLOUDS
OBJECT_BOUNDING_BOXES = base_augmentation_layer_3d.OBJECT_BOUNDING_BOXES


@keras_cv_export("keras_cv.layers.GroupPointsByBoundingBoxes")
class GroupPointsByBoundingBoxes(
    base_augmentation_layer_3d.BaseAugmentationLayer3D
):
    """A preprocessing layer which groups point clouds based on bounding boxes
    during training.

    This layer will group point clouds based on bounding boxes and generate
    OBJECT_POINT_CLOUDS and OBJECT_BOUNDING_BOXES tensors.

    Input shape:
      point_clouds: 3D (multi frames) float32 Tensor with shape
        [num of frames, num of points, num of point features].
        The first 5 features are [x, y, z, class, range].
      bounding_boxes: 3D (multi frames) float32 Tensor with shape
        [num of frames, num of boxes, num of box features]. Boxes are expected
        to follow the CENTER_XYZ_DXDYDZ_PHI format. Refer to
        https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box_3d/formats.py

    Output shape:
      A dictionary of Tensors with the same shape as input Tensors and two
      additional items for OBJECT_POINT_CLOUDS (shape [num of frames, num of
      valid boxes, max num of points, num of point features]) and
      OBJECT_BOUNDING_BOXES (shape [num of frames, num of valid boxes, num of
      box features]).

    Arguments:
      label_index: An optional int scalar sets the target object index.
        Bounding boxes and corresponding point clouds with box class ==
        label_index will be saved as OBJECT_BOUNDING_BOXES and
        OBJECT_POINT_CLOUDS. If label index is None, all valid bounding boxes
        (box class !=0) are used.
      min_points_per_bounding_boxes: A int scalar sets the min number of points
        in a bounding box. If a bounding box contains less than
        min_points_per_bounding_boxes, the bounding box is filtered out.
      max_points_per_bounding_boxes: A int scalar sets the max number of points
        in a bounding box. All the object point clouds will be padded or trimmed
        to the same shape, where the number of points dimension is
        max_points_per_bounding_boxes.
    """

    def __init__(
        self,
        label_index=None,
        min_points_per_bounding_boxes=0,
        max_points_per_bounding_boxes=2000,
        **kwargs
    ):
        super().__init__(**kwargs)

        if label_index and label_index < 0:
            raise ValueError("label_index must be >=0 or None.")
        if min_points_per_bounding_boxes < 0:
            raise ValueError("min_points_per_bounding_boxes must be >=0.")
        if max_points_per_bounding_boxes < 0:
            raise ValueError("max_points_per_bounding_boxes must be >=0.")
        if min_points_per_bounding_boxes > max_points_per_bounding_boxes:
            raise ValueError(
                "max_paste_bounding_boxes must be >= "
                "min_points_per_bounding_boxes."
            )

        self._label_index = label_index
        self._min_points_per_bounding_boxes = min_points_per_bounding_boxes
        self._max_points_per_bounding_boxes = max_points_per_bounding_boxes
        self._auto_vectorize = False

    def get_config(self):
        return {
            "label_index": self._label_index,
            "min_points_per_bounding_boxes": self._min_points_per_bounding_boxes,  # noqa: E501
            "max_points_per_bounding_boxes": self._max_points_per_bounding_boxes,  # noqa: E501
        }

    def augment_point_clouds_bounding_boxes(
        self, point_clouds, bounding_boxes, **kwargs
    ):
        if self._label_index:
            bounding_boxes_mask = (
                bounding_boxes[0, :, CENTER_XYZ_DXDYDZ_PHI.CLASS]
                == self._label_index
            )
            object_bounding_boxes = tf.boolean_mask(
                bounding_boxes, bounding_boxes_mask, axis=1
            )
        else:
            bounding_boxes_mask = (
                bounding_boxes[0, :, CENTER_XYZ_DXDYDZ_PHI.CLASS] > 0.0
            )
            object_bounding_boxes = tf.boolean_mask(
                bounding_boxes, bounding_boxes_mask, axis=1
            )

        points_in_bounding_boxes = is_within_box3d(
            point_clouds[:, :, :3], object_bounding_boxes[:, :, :7]
        )
        # Filter bounding boxes using the current frame.
        # [num_boxes]
        min_points_filter = (
            tf.reduce_sum(
                tf.cast(points_in_bounding_boxes[0], dtype=tf.int32), axis=0
            )
            >= self._min_points_per_bounding_boxes
        )

        object_bounding_boxes = tf.boolean_mask(
            object_bounding_boxes, min_points_filter, axis=1
        )

        points_in_bounding_boxes = tf.boolean_mask(
            points_in_bounding_boxes, min_points_filter, axis=2
        )
        # [num of frames, num of boxes, num of points].
        points_in_bounding_boxes = tf.transpose(
            points_in_bounding_boxes, [0, 2, 1]
        )
        points_in_bounding_boxes = tf.cast(points_in_bounding_boxes, tf.int32)
        sort_valid_index = tf.argsort(
            points_in_bounding_boxes, axis=-1, direction="DESCENDING"
        )
        sort_valid_mask = tf.gather(
            points_in_bounding_boxes, sort_valid_index, axis=2, batch_dims=2
        )[:, :, : self._max_points_per_bounding_boxes]
        # [num of frames, num of boxes, self._max_points_per_bounding_boxes, num
        # of point features].
        object_point_clouds = point_clouds[:, tf.newaxis, :, :]
        num_valid_bounding_boxes = tf.shape(object_bounding_boxes)[1]
        object_point_clouds = tf.tile(
            object_point_clouds, [1, num_valid_bounding_boxes, 1, 1]
        )
        object_point_clouds = tf.gather(
            object_point_clouds, sort_valid_index, axis=2, batch_dims=2
        )[:, :, : self._max_points_per_bounding_boxes, :]

        object_point_clouds = tf.where(
            sort_valid_mask[:, :, :, tf.newaxis] > 0, object_point_clouds, 0.0
        )

        return (
            object_point_clouds,
            object_bounding_boxes,
        )

    def augment_point_clouds_bounding_boxes_v2(
        self, point_clouds, bounding_boxes, **kwargs
    ):
        if self._label_index:
            bounding_boxes_mask = (
                bounding_boxes[0, :, CENTER_XYZ_DXDYDZ_PHI.CLASS]
                == self._label_index
            )
            object_bounding_boxes = tf.boolean_mask(
                bounding_boxes, bounding_boxes_mask, axis=1
            )
        else:
            bounding_boxes_mask = (
                bounding_boxes[0, :, CENTER_XYZ_DXDYDZ_PHI.CLASS] > 0.0
            )
            object_bounding_boxes = tf.boolean_mask(
                bounding_boxes, bounding_boxes_mask, axis=1
            )

        # [frames, num_boxes, ragged_points]
        points_in_bounding_boxes = group_points_by_boxes(
            point_clouds[:, :, :3], object_bounding_boxes[:, :, :7]
        )
        # Filter bounding boxes using the current frame.
        # [num_boxes]
        min_points_filter = (
            points_in_bounding_boxes.row_lengths(-1)
            >= self._min_points_per_bounding_boxes
        )

        # [frames, num_valid_boxes, box_feature]
        object_bounding_boxes = tf.ragged.boolean_mask(
            object_bounding_boxes, min_points_filter
        )
        # [frames, num_valid_boxes, ragged_points]
        points_in_bounding_boxes = tf.ragged.boolean_mask(
            points_in_bounding_boxes, min_points_filter
        )
        # point_clouds: [frames, num_points, point_feature]
        # object_point_clouds: [frames, num_valid_boxes, ragged_points,
        #   point_feature]
        object_point_clouds = tf.gather(
            point_clouds, points_in_bounding_boxes, axis=1, batch_dims=1
        )

        return (object_point_clouds, object_bounding_boxes)

    def _augment(self, inputs):
        result = inputs
        point_clouds = inputs[POINT_CLOUDS]
        bounding_boxes = inputs[BOUNDING_BOXES]

        transformation = self.get_random_transformation(
            point_clouds=point_clouds,
            bounding_boxes=bounding_boxes,
        )
        (
            object_point_clouds,
            object_bounding_boxes,
        ) = self.augment_point_clouds_bounding_boxes(
            point_clouds,
            bounding_boxes=bounding_boxes,
            transformation=transformation,
        )

        result.update(
            {
                OBJECT_POINT_CLOUDS: object_point_clouds,
                OBJECT_BOUNDING_BOXES: object_bounding_boxes,
            }
        )
        return result

    def call(self, inputs):
        # TODO(ianstenbit): Support the model input format.
        point_clouds = inputs[POINT_CLOUDS]
        bounding_boxes = inputs[BOUNDING_BOXES]
        if point_clouds.shape.rank == 3 and bounding_boxes.shape.rank == 3:
            return self._augment(inputs)
        elif point_clouds.shape.rank == 4 and bounding_boxes.shape.rank == 4:
            batch = point_clouds.get_shape().as_list()[0]
            object_point_clouds_list = []
            object_bounding_boxes_list = []
            for i in range(batch):
                (
                    object_point_clouds,
                    object_bounding_boxes,
                ) = self.augment_point_clouds_bounding_boxes(
                    inputs[POINT_CLOUDS][i], inputs[BOUNDING_BOXES][i]
                )
                object_point_clouds_list += [object_point_clouds]
                object_bounding_boxes_list += [object_bounding_boxes]
            # object_point_clouds shape [num of frames, num of valid boxes,
            # max num of points, num of point features].
            inputs[OBJECT_POINT_CLOUDS] = tf.concat(
                object_point_clouds_list, axis=-3
            )
            # object_bounding_boxes shape [num of frames, num of valid
            # boxes, num of box features].
            inputs[OBJECT_BOUNDING_BOXES] = tf.concat(
                object_bounding_boxes_list, axis=-2
            )
            return inputs
        else:
            raise ValueError(
                "Point clouds augmentation layers are expecting inputs "
                "point clouds and bounding boxes to be rank 3D (Frame, "
                "Point, Feature) or 4D (Batch, Frame, Point, Feature) "
                "tensors. Got shape: {} and {}".format(
                    point_clouds.shape, bounding_boxes.shape
                )
            )
