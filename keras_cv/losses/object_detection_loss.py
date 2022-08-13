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

import tensorflow as tf


class ObjectDetectionLoss(tf.keras.losses.Loss):
    """ObjectDetectionLoss formats object detection predictions for loss computation.

    For more context on how to use this loss, please see:
    `examples/applications/object_detection/retina_net/basic/pascal_voc/train.py`

    Args:
        classes: number of classes to use in loss calculation.
        classification_loss: `keras.losses.Loss` to apply to the classification
            predictions made by your object detection model.
        box_loss: `keras.losses.Loss` to apply to the bounding boxes.  This loss will
            receive a Tensor of shape [None, None, 4] for the y_true and y_pred.
    """

    def __init__(
        self, classes, classification_loss, box_loss, reduction="auto", **kwargs
    ):
        super().__init__(**kwargs, reduction=reduction)

        # TODO(lukewood): can we just update reductions to 'none'?
        if classification_loss.reduction != "none":
            raise ValueError(
                "When using `keras_cv.losses.ObjectDetectionLoss()`, "
                "please pass `reduction='none'` to both `classification_loss` and "
                "`box_loss` and pass `reduction` to `keras_cv.losses.ObjectDetectionLoss()` "
                "to handle reduction.  Received "
                f"classification_loss.reduction={classification_loss.reduction}"
            )

        if box_loss.reduction != "none":
            raise ValueError(
                "When using `keras_cv.losses.ObjectDetectionLoss()`, "
                "please pass `reduction='none'` to both `classification_loss` and "
                "`box_loss` and pass `reduction` to `keras_cv.losses.ObjectDetectionLoss()` "
                "to handle reduction.  Received "
                f"box_loss.reduction={box_loss.reduction}"
            )

        self.classes = classes
        self.classification_loss = classification_loss
        self.box_loss = box_loss

    def call(self, y_true, y_pred):
        box_labels = y_true[:, :, :4]
        box_predictions = y_pred[:, :, :4]

        if y_true.shape[-1] != 5:
            raise ValueError(
                "y_true should have shape (None, None, 5).  Got "
                f"y_true.shape={tuple(y_true.shape)}"
            )

        if y_pred.shape[-1] != self.classes + 4:
            raise ValueError(
                "y_pred should have shape (None, None, classes + 4). "
                f"Got y_pred.shape={tuple(y_pred.shape)}.  Does your model's `classes` "
                "parameter match your losses `classes` parameter?"
            )

        cls_labels = tf.one_hot(
            tf.cast(y_true[:, :, 4], dtype=tf.int32),
            depth=self.classes,
            dtype=tf.float32,
        )
        cls_predictions = y_pred[:, :, 4:]

        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)

        classification_loss = self.classification_loss(cls_labels, cls_predictions)
        box_loss = self.box_loss(box_labels, box_predictions)

        classification_loss = tf.where(
            tf.equal(ignore_mask, 1.0), 0.0, classification_loss
        )
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        classification_loss = tf.math.divide_no_nan(
            tf.reduce_sum(classification_loss, axis=-1), normalizer
        )
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)

        return box_loss + classification_loss
