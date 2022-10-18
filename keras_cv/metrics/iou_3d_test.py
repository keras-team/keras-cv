# Copyright 2022 The KerasCV Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for IoU3D using custom op."""
import math
import os

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from keras_cv.metrics import IoU3D


class IoU3DTest(tf.test.TestCase):
    @pytest.mark.skipif(
        "TEST_CUSTOM_OPS" not in os.environ or os.environ["TEST_CUSTOM_OPS"] != "true",
        reason="Requires binaries compiled from source",
    )
    def testOpCall(self):
        # Predicted boxes:
        # 0: a 2x2x2 box centered at 0,0,0, rotated 0 degrees
        # 1: a 2x2x2 box centered at 1,1,1, rotated 135 degrees
        # Ground Truth boxes:
        # 0: a 2x2x2 box centered at 1,1,1, rotated 45 degrees (idential to predicted box 1)
        # 1: a 2x2x2 box centered at 1,1,1, rotated 0 degrees
        box_preds = [[0, 0, 0, 2, 2, 2, 0], [1, 1, 1, 2, 2, 2, 3 * math.pi / 4]]
        box_gt = [[1, 1, 1, 2, 2, 2, math.pi / 4], [1, 1, 1, 2, 2, 2, 0]]

        # Predicted box 0 and ground truth box 0 overlap by 1/8th of the box.
        # Therefore, IiU is 1/15
        # Predicted box 1 shares an origin with ground truth box 1, but is rotated by 135 degrees.
        # Their IoU can be reduced to that of two overlapping squares that share a center with
        # the same offset of 135 degrees, which reduces to the square root of 0.5.
        expected_ious = [1 / 15, 0.5**0.5]
        expected_mean_iou = np.mean(expected_ious)

        iou_3d = IoU3D()
        iou_3d.update_state(box_preds, box_gt)

        self.assertAllClose(iou_3d.result(), expected_mean_iou)

    @pytest.mark.skipif(
        "TEST_CUSTOM_OPS" not in os.environ or os.environ["TEST_CUSTOM_OPS"] != "true",
        reason="Requires binaries compiled from source",
    )
    def test_runs_inside_model(self):
        i = keras.layers.Input((None, None, 6))
        model = keras.Model(i, i)

        iou_3d = IoU3D()

        # Two pairs of boxes with 100% IoU
        y_true = np.array(
            [[0, 0, 0, 2, 2, 2, 0], [1, 1, 1, 2, 2, 2, 3 * math.pi / 4]]
        ).astype(np.float32)
        y_pred = np.array(
            [[0, 0, 0, 2, 2, 2, math.pi], [1, 1, 1, 2, 2, 2, math.pi / 4]]
        ).astype(np.float32)

        model.compile(metrics=[iou_3d])
        model.evaluate(y_pred, y_true)

        self.assertAllEqual(iou_3d.result(), 1.0)


if __name__ == "__main__":
    tf.test.main()
