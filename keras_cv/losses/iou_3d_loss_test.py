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
"""Tests for IoU3DLoss using custom op."""
import math
import os

import pytest
import tensorflow as tf

from keras_cv.losses import IoU3DLoss


class IoU3DTest(tf.test.TestCase):
    @pytest.mark.skipif(
        "SKIP_CUSTOM_OPS" in os.environ and os.environ["SKIP_CUSTOM_OPS"] == "true",
        reason="Requires binaries compiled from source",
    )
    def testOpCall(self):
        box_preds = [[0, 0, 0, 2, 2, 2, 0], [1, 1, 1, 2, 2, 2, 3 * math.pi / 4]]
        box_gt = [[1, 1, 1, 2, 2, 2, math.pi / 4], [1, 1, 1, 2, 2, 2, 0]]

        self.assertAllClose(
            IoU3DLoss(box_preds, box_gt), [[2 / 30, 2 / 30], [1, 0.707107]]
        )


if __name__ == "__main__":
    tf.test.main()
