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


class RetinaNetLoss(tf.keras.losses.Loss):
    def __init__(
        self, num_classes, alpha=0.25, gamma=2.0, delta=1.0, reduction="auto", **kwargs
    ):
        super().__init__(**kwargs, reduction=reduction)
        self.num_classes = num_classes
        self.classification_loss = RetinaNetClassificationLoss(
            alpha=alpha, gamma=gamma, reduction=reduction
        )
        self.box_loss = RetinaNetBoxLoss(delta=delta, reduction=reduction)

    def call(self, y_true, y_pred):
        box_labels = y_true[:, :, :4]
        box_predictions = y_pred[:, :, :4]

        if y_true.shape[-1] != 5:
            raise ValueError(
                "y_true should have shape [None, None, 5].  Got "
                f"y_true.shape={y_true.shape}"
            )

        if y_pred.shape[-1] != self.num_classes + 4:
            raise ValueError(
                "y_pred should have shape [None, None, 4+num_classes]. "
                f"Got y_pred.shape={y_pred.shape}."
            )

        cls_labels = tf.one_hot(
            tf.cast(y_true[:, :, 4], dtype=tf.int32),
            depth=self.num_classes,
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
        return classification_loss + box_loss


# --- Implementing Smooth L1 loss and Focal Loss as keras custom losses ---
class RetinaNetBoxLoss(tf.keras.losses.Loss):
    """Implements Smooth L1 loss"""

    def __init__(self, delta, reduction="none", **kwargs):
        super().__init__(reduction=reduction, **kwargs)
        self._delta = delta

    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference**2
        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetClassificationLoss(tf.keras.losses.Loss):
    """Implements Focal loss"""

    def __init__(self, alpha, gamma, reduction="none", **kwargs):
        super().__init__(reduction=reduction, **kwargs)
        self._alpha = alpha
        self._gamma = gamma

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)
