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
import numpy as np
from keras_cv.losses.hausdorff import Hausdorff


class HausdorffTest(tf.test.TestCase):
    def test_hausdorff_loss_diff(self):
        y_true = np.array(
                            [[1, 1, 0],
                             [1, 1, 1],
                             [1, 1, 1]])

        y_pred = np.array(
                            [[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]])

        hausdorff_distance = Hausdorff()
        self.assertNotEqual(hausdorff_distance(y_true, y_pred).numpy(), 0.0)

    def test_hausdorff_loss_same(self):
        y_true = np.array(
                            [[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]])

        y_pred = np.array(
                            [[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]])

        hausdorff_distance = Hausdorff()
        self.assertEqual(hausdorff_distance(y_true, y_pred).numpy(), 0.0)
