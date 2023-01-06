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

from keras_cv.losses import BinaryCenterNetCrossentropy


class BinaryCenterNetCrossentropyTest(tf.test.TestCase):
    def test_output_shape(self):
        y_true = tf.cast(
            tf.random.uniform(shape=[2, 5], minval=0, maxval=2, dtype=tf.int32),
            tf.float32,
        )
        y_pred = tf.random.uniform(shape=[2, 5], minval=0, maxval=1, dtype=tf.float32)

        focal_loss = BinaryCenterNetCrossentropy(reduction="sum")

        self.assertAllEqual(focal_loss(y_true, y_pred).shape, [])

    def test_output_shape_reduction_none(self):
        y_true = tf.cast(
            tf.random.uniform(shape=[2, 5], minval=0, maxval=2, dtype=tf.int32),
            tf.float32,
        )
        y_pred = tf.random.uniform(shape=[2, 5], minval=0, maxval=1, dtype=tf.float32)

        focal_loss = BinaryCenterNetCrossentropy(reduction="none")

        self.assertAllEqual(
            [2, 5],
            focal_loss(y_true, y_pred).shape,
        )

    def test_output_with_pos_label_pred(self):
        y_true = tf.constant([1.0])
        y_pred = tf.constant([1.0])
        focal_loss = BinaryCenterNetCrossentropy(reduction="sum")
        self.assertAllClose(0.0, focal_loss(y_true, y_pred))

    def test_output_with_pos_label_neg_pred(self):
        y_true = tf.constant([1.0])
        y_pred = tf.constant([np.exp(-1.0)])
        focal_loss = BinaryCenterNetCrossentropy(reduction="sum")
        # (1-1/e)^2 * log(1/e)
        self.assertAllClose(np.square(1 - np.exp(-1.0)), focal_loss(y_true, y_pred))

    def test_output_with_neg_label_pred(self):
        y_true = tf.constant([0.0])
        y_pred = tf.constant([0.0])
        focal_loss = BinaryCenterNetCrossentropy(reduction="sum")
        self.assertAllClose(0.0, focal_loss(y_true, y_pred))

    def test_output_with_neg_label_pos_pred(self):
        y_true = tf.constant([0.0])
        y_pred = tf.constant([1.0 - np.exp(-1.0)])
        focal_loss = BinaryCenterNetCrossentropy(reduction="sum")
        # (1-0)^4 * (1-1/e)^2 * log(1/e)
        self.assertAllClose(np.square(1 - np.exp(-1.0)), focal_loss(y_true, y_pred))

    def test_output_with_weak_label_pos_pred(self):
        y_true = tf.constant([0.5])
        y_pred = tf.constant([1.0 - np.exp(-1.0)])
        focal_loss = BinaryCenterNetCrossentropy(beta=2.0, reduction="sum")
        # (1-0.5)^2 * (1-1/e)^2 * log(1/e)
        self.assertAllClose(
            0.25 * np.square(1 - np.exp(-1.0)), focal_loss(y_true, y_pred)
        )

    def test_output_with_sample_weight(self):
        y_true = tf.constant([0.0])
        y_pred = tf.constant([1.0 - np.exp(-1.0)])
        sample_weight = tf.constant([0.5])
        focal_loss = BinaryCenterNetCrossentropy(reduction="sum")
        # (1-0)^4 * (1-1/e)^2 * log(1/e)
        self.assertAllClose(
            0.5 * np.square(1 - np.exp(-1.0)),
            focal_loss(y_true, y_pred, sample_weight=sample_weight),
        )
