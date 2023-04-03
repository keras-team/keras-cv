# Copyright 2023 The KerasCV Authors
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

import math

import tensorflow as tf
from tensorflow import keras

from keras_cv.bounding_box_3d import CENTER_XYZ_DXDYDZ_PHI


def l1(y_true, y_pred, sigma=9.0):
    """Computes element-wise l1 loss."""

    absolute_difference = tf.abs(y_pred - y_true)
    loss = tf.where(
        absolute_difference < 1.0 / sigma,
        0.5 * sigma * absolute_difference**2,
        absolute_difference - 0.5 / sigma,
    )

    return tf.reduce_sum(loss, axis=-1)


class Box3DRegressionLoss(keras.losses.Loss):
    """Implements a bin-based box regression loss for 3D bounding boxes.

    Reference: https://arxiv.org/abs/1812.04244

    Box3DRegressionLoss uses L1 loss on the individual components of boxes, with
    the exception of the bin-based heading component of each box, where the bin
    indicator outputs use cross entropy loss, and the heading residual uses L1
    loss.

    Ground truth boxes are expected to follow the CENTER_XYZ_DXDYDZ_PHI format.
    Refer to https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box_3d/formats.py
    for more details on supported bounding box formats.

    Box predictions are expected to be in CenterPillar heatmap-encoded format.

    Args:
        num_heading_bins: int, number of bins used for predicting box heading.
        anchor_size: list of 3 ints, anchor sizes for the x, y, and z axes.
    """

    def __init__(self, num_heading_bins, anchor_size, **kwargs):
        super().__init__(**kwargs)
        self.num_heading_bins = num_heading_bins
        self.anchor_size = anchor_size

    def heading_regression_loss(self, heading_true, heading_pred):
        # Set the heading to within 0 -> 2pi
        heading_true = tf.math.floormod(heading_true, 2 * math.pi)

        # Divide 2pi into bins. shifted by 0.5 * angle_per_class.
        angle_per_class = (2 * math.pi) / self.num_heading_bins
        shift_angle = tf.math.floormod(
            heading_true + angle_per_class / 2, 2 * math.pi
        )

        heading_bin_label_float = tf.math.floordiv(shift_angle, angle_per_class)
        heading_bin_label = tf.cast(heading_bin_label_float, dtype=tf.int32)
        heading_res_label = shift_angle - (
            heading_bin_label_float * angle_per_class + angle_per_class / 2.0
        )
        heading_res_norm_label = heading_res_label / (angle_per_class / 2.0)

        heading_bin_one_hot = tf.one_hot(
            heading_bin_label, self.num_heading_bins
        )
        loss_heading_bin = tf.nn.softmax_cross_entropy_with_logits(
            labels=heading_bin_one_hot,
            logits=heading_pred[..., : self.num_heading_bins],
        )
        loss_heading_res = l1(
            tf.reduce_sum(
                heading_pred[..., self.num_heading_bins :]
                * heading_bin_one_hot,
                axis=-1,
                keepdims=True,
            ),
            heading_res_norm_label[..., tf.newaxis],
        )

        return loss_heading_bin + loss_heading_res

    def regression_loss(self, y_true, y_pred):
        position_loss = l1(y_true[:, :3], y_pred[:, :3])

        heading_loss = self.heading_regression_loss(
            y_true[:, CENTER_XYZ_DXDYDZ_PHI.PHI], y_pred[:, 3:-3]
        )

        # Size loss
        size_norm_label = y_true[:, 3:6] / self.anchor_size
        size_norm_pred = y_pred[:, -3:] + 1.0
        size_loss = l1(size_norm_pred, size_norm_label)

        # TODO(ianstenbit): Add IoU3D Loss.

        return position_loss + heading_loss + size_loss

    def call(self, y_true, y_pred):
        return tf.map_fn(
            lambda y_true_and_pred: self.regression_loss(
                y_true_and_pred[0], y_true_and_pred[1]
            ),
            (y_true, y_pred),
            tf.TensorSpec((y_pred.shape[1])),
        )

    def get_config(self):
        config = {
            "num_heading_bins": self.num_heading_bins,
            "anchor_size": self.anchor_size,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
