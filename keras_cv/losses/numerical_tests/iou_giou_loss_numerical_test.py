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

import numpy as np
import tensorflow as tf

import keras_cv


def tf_calculate_giou(b1, b2, mode: str = "giou") -> tf.Tensor:
    """
    Args:
        b1: bounding box. The coordinates of the each bounding box in boxes are
            encoded as [y_min, x_min, y_max, x_max].
        b2: the other bounding box. The coordinates of the each bounding box
            in boxes are encoded as [y_min, x_min, y_max, x_max].
        mode: one of ['giou', 'iou'], decided to calculate GIoU or IoU loss.
    Returns:
        GIoU loss float `Tensor`.
    """
    zero = tf.convert_to_tensor(0.0, b1.dtype)
    b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)
    b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
    b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
    b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
    b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
    b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height

    intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
    intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
    intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
    intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
    intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
    intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height

    union_area = b1_area + b2_area - intersect_area
    iou = tf.math.divide_no_nan(intersect_area, union_area)
    if mode == "iou":
        return iou

    enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
    enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
    enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
    enclose_xmax = tf.maximum(b1_xmax, b2_xmax)
    enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
    enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
    enclose_area = enclose_width * enclose_height
    giou = iou - tf.math.divide_no_nan((enclose_area - union_area), enclose_area)
    return giou


def tf_addons_giou_loss(y_true, y_pred, mode: str = "giou") -> tf.Tensor:
    """Implements the GIoU loss function.
    GIoU loss was first introduced in the
    [Generalized Intersection over Union:
    A Metric and A Loss for Bounding Box Regression]
    (https://giou.stanford.edu/GIoU.pdf).
    GIoU is an enhancement for models which use IoU in object detection.
    Args:
        y_true: true targets tensor. The coordinates of the each bounding
            box in boxes are encoded as [y_min, x_min, y_max, x_max].
        y_pred: predictions tensor. The coordinates of the each bounding
            box in boxes are encoded as [y_min, x_min, y_max, x_max].
        mode: one of ['giou', 'iou'], decided to calculate GIoU or IoU loss.
    Returns:
        GIoU loss float `Tensor`.
    """
    if mode not in ["giou", "iou"]:
        raise ValueError("Value of mode should be 'iou' or 'giou'")
    y_pred = tf.convert_to_tensor(y_pred)
    if not y_pred.dtype.is_floating:
        y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, y_pred.dtype)
    giou = tf.squeeze(tf_calculate_giou(y_pred, y_true, mode))

    return 1 - giou


class IoULossAddonsComparisonTest(tf.test.TestCase):
    def test_iou_output_value(self):
        y_true = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 2, 3],
                [4, 5, 6, 6],
                [2, 2, 3, 3],
            ]
        )

        y_pred = np.array(
            [
                [0, 0, 5, 6],
                [0, 0, 7, 3],
                [7, 8, 9, 10],
                [2, 1, 3, 3],
            ]
        )

        iou_loss = keras_cv.losses.IoULoss(
            bounding_box_format="yxyx",
            mode="linear",
            reduction="none",
            axis="no_reduction",
        )
        self.assertAllClose(
            iou_loss(y_true, y_pred),
            tf_addons_giou_loss(y_true, y_pred, mode="iou"),
        )

    def test_giou_output_value(self):
        y_true = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 2, 3],
                [4, 5, 6, 6],
                [2, 2, 3, 3],
            ]
        )

        y_pred = np.array(
            [
                [0, 0, 5, 6],
                [0, 0, 7, 3],
                [7, 8, 9, 10],
                [2, 1, 3, 3],
            ]
        )

        giou_loss = keras_cv.losses.GIoULoss(
            bounding_box_format="yxyx", reduction="none", axis="no_reduction"
        )
        self.assertAllClose(
            giou_loss(y_true, y_pred), tf_addons_giou_loss(y_true, y_pred)
        )
