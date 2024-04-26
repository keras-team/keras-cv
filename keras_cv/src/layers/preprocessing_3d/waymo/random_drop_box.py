# Copyright 2022 Waymo LLC.
#
# Licensed under the terms in https://github.com/keras-team/keras-cv/blob/master/keras_cv/layers/preprocessing_3d/waymo/LICENSE  # noqa: E501

import tensorflow as tf

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.layers.preprocessing_3d import base_augmentation_layer_3d
from keras_cv.src.point_cloud import is_within_any_box3d

POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES
BOX_LABEL_INDEX = base_augmentation_layer_3d.BOX_LABEL_INDEX


@keras_cv_export("keras_cv.layers.RandomDropBox")
class RandomDropBox(base_augmentation_layer_3d.BaseAugmentationLayer3D):
    """A preprocessing layer which randomly drops object bounding boxes and
    points during training.

    This layer will randomly drop object point clouds and bounding boxes. Number
    of dropped bounding boxes is sampled uniformly sampled between 0 and
    max_drop_bounding_boxes. If label_index is set, only bounding boxes with box
    class == label_index will be sampled and dropped; otherwise, all valid
    bounding boxes (box class > 0) will be sampled and dropped.

    Input shape:
      point_clouds: 3D (multi frames) float32 Tensor with shape
        [num of frames, num of points, num of point features].
        The first 5 features are [x, y, z, class, range].
      bounding_boxes: 3D (multi frames) float32 Tensor with shape
        [num of frames, num of boxes, num of box features].
        The first 8 features are [x, y, z, dx, dy, dz, phi, box class].


    Output shape:
      A tuple of two Tensors (point_clouds, bounding_boxes) with the same shape
      as input Tensors.

    Arguments:
      max_drop_bounding_boxes: A int non-negative scalar sets the maximum number
        of dropped bounding boxes. Do not drop any bounding boxes when
        max_drop_bounding_boxes = 0.
      label_index: An optional int scalar sets the target object index.
        If label index is set, randomly drop bounding boxes, where box
        class == label_index.
        If label index is None, randomly drop bounding boxes, where box
        class > 0.

    """

    def __init__(self, max_drop_bounding_boxes, label_index=None, **kwargs):
        super().__init__(**kwargs)
        self.auto_vectorize = False
        if label_index and label_index < 0:
            raise ValueError("label_index must be >=0 or None.")
        if max_drop_bounding_boxes < 0:
            raise ValueError("max_drop_bounding_boxes must be >=0.")
        self._label_index = label_index
        self._max_drop_bounding_boxes = max_drop_bounding_boxes

    def get_config(self):
        return {
            "label_index": self._label_index,
            "max_drop_bounding_boxes": self._max_drop_bounding_boxes,
        }

    def get_random_transformation(self, point_clouds, bounding_boxes, **kwargs):
        if not self._max_drop_bounding_boxes:
            return {}
        del point_clouds
        if self._label_index:
            selected_boxes_mask = (
                bounding_boxes[0, :, BOX_LABEL_INDEX] == self._label_index
            )
        else:
            selected_boxes_mask = tf.math.greater(
                bounding_boxes[0, :, BOX_LABEL_INDEX], 0
            )
        max_drop_bounding_boxes = tf.random.uniform(
            (), maxval=self._max_drop_bounding_boxes, dtype=tf.int32
        )
        # Randomly remove max_drop_bounding_boxes number of bounding boxes.
        num_bounding_boxes = bounding_boxes.get_shape().as_list()[1]
        random_scores_for_selected_boxes = tf.random.uniform(
            shape=[num_bounding_boxes]
        )
        random_scores_for_selected_boxes = tf.where(
            selected_boxes_mask, random_scores_for_selected_boxes, 0.0
        )
        topk, _ = tf.math.top_k(
            random_scores_for_selected_boxes, k=max_drop_bounding_boxes + 1
        )
        drop_bounding_boxes_mask = tf.math.greater(
            random_scores_for_selected_boxes, topk[-1]
        )

        # Only drop selected bounding boxes.
        drop_bounding_boxes_mask &= selected_boxes_mask
        return {
            "drop_bounding_boxes_mask": drop_bounding_boxes_mask,
        }

    def augment_point_clouds_bounding_boxes(
        self, point_clouds, bounding_boxes, transformation, **kwargs
    ):
        if not self._max_drop_bounding_boxes:
            return (point_clouds, bounding_boxes)
        drop_bounding_boxes_mask = transformation["drop_bounding_boxes_mask"]

        drop_bounding_boxes = tf.boolean_mask(
            bounding_boxes, drop_bounding_boxes_mask, axis=1
        )

        drop_points_mask = is_within_any_box3d(
            point_clouds[..., :3], drop_bounding_boxes[..., :7], keepdims=True
        )
        return (
            tf.where(~drop_points_mask, point_clouds, 0.0),
            tf.where(
                ~drop_bounding_boxes_mask[tf.newaxis, :, tf.newaxis],
                bounding_boxes,
                0.0,
            ),
        )
