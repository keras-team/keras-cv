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


def bbox_iou(box1, box2, eps=1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = tf.split(box1, 4, axis=-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = tf.split(box2, 4, axis=-1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = tf.math.maximum(
        tf.math.minimum(b1_x2, b2_x2) - tf.math.maximum(b1_x1, b2_x1), 0
    ) * tf.math.maximum(
        tf.math.minimum(b1_y2, b2_y2) - tf.math.maximum(b1_y1, b2_y1), 0
    )

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    cw = tf.math.maximum(b1_x2, b2_x2) - tf.math.minimum(
        b1_x1, b2_x1
    )  # convex (smallest enclosing box) width
    ch = tf.math.maximum(b1_y2, b2_y2) - tf.math.minimum(
        b1_y1, b2_y1
    )  # convex height
    c2 = cw**2 + ch**2 + eps  # convex diagonal squared
    rho2 = (
        (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
        + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
    ) / 4  # center dist ** 2
    v = tf.pow((4 / math.pi**2) * (tf.atan(w2 / h2) - tf.atan(w1 / h1)), 2)
    alpha = v / (v - iou + (1 + eps))

    return iou - (rho2 / c2 + v * alpha)


# TODO(ianstenbit): Use keras_cv.losses.IoULoss instead
# (It needs to support CIoU as well as dynamic number of boxes)
class YOLOV8IoULoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        # IoU loss
        iou = bbox_iou(y_pred, y_true)

        return 1.0 - iou
