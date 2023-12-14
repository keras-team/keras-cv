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
from absl.testing import parameterized
from tensorflow import keras

from keras_cv.backend import ops
from keras_cv.losses import FocalLoss
from keras_cv.tests.test_case import TestCase


class ModelGardenFocalLoss(keras.losses.Loss):
    def __init__(
        self, alpha, gamma, reduction=keras.losses.Reduction.AUTO, name=None
    ):
        self._alpha = alpha
        self._gamma = gamma
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        with tf.name_scope("focal_loss"):
            y_true = tf.cast(y_true, dtype=tf.float32)
            y_pred = tf.cast(y_pred, dtype=tf.float32)
            positive_label_mask = tf.equal(y_true, 1.0)
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=y_true, logits=y_pred
            )
            probs = tf.sigmoid(y_pred)
            probs_gt = tf.where(positive_label_mask, probs, 1.0 - probs)
            # With small gamma, the implementation could produce NaN during back
            # prop.
            modulator = tf.pow(1.0 - probs_gt, self._gamma)
            loss = modulator * cross_entropy
            weighted_loss = tf.where(
                positive_label_mask,
                self._alpha * loss,
                (1.0 - self._alpha) * loss,
            )

        return weighted_loss


class FocalLossModelGardenComparisonTest(TestCase):
    @parameterized.named_parameters(
        ("sum", "sum"),
    )
    def test_model_garden_implementation_has_same_outputs(self, reduction):
        focal_loss = FocalLoss(
            alpha=0.25, gamma=2.0, from_logits=False, reduction=reduction
        )
        model_garden_focal_loss = ModelGardenFocalLoss(
            alpha=0.25, gamma=2.0, reduction=reduction
        )

        for _ in range(10):
            y_true = np.random.randint(size=(200,), low=0, high=10)
            y_true = tf.one_hot(y_true, depth=10)
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.random.uniform((200, 10), dtype=tf.float32)
            self.assertAllClose(
                ops.convert_to_numpy(focal_loss(y_true, tf.sigmoid(y_pred))),
                ops.convert_to_numpy(model_garden_focal_loss(y_true, y_pred)),
            )
