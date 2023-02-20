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

from keras_cv.layers.object_detection_3d import voxel_utils


class PadOrTrimToTest(tf.test.TestCase):
    """Tests for pad_or_trim_to, branched from https://github.com/tensorflow/lingvo/blob/master/lingvo/core/py_utils_test.py."""

    def test_2D_constant_shape_pad(self):
        x = tf.random.normal(shape=(3, 3), seed=123456)
        shape = [4, 6]
        padded_x_right = voxel_utils._pad_or_trim_to(x, shape, pad_val=0)
        padded_x_left = voxel_utils._pad_or_trim_to(
            x, shape, pad_val=0, pad_after_contents=False
        )
        self.assertEqual(padded_x_right.shape.as_list(), [4, 6])
        self.assertEqual(padded_x_left.shape.as_list(), [4, 6])
        real_x_right, real_x_left = self.evaluate(
            [padded_x_right, padded_x_left]
        )
        expected_x_right = [
            [0.38615, 2.975221, -0.852826, 0.0, 0.0, 0.0],
            [-0.571142, -0.432439, 0.413158, 0.0, 0.0, 0.0],
            [0.255314, -0.985647, 1.461641, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        self.assertAllClose(expected_x_right, real_x_right)
        expected_x_left = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.38615, 2.975221, -0.852826],
            [0.0, 0.0, 0.0, -0.571142, -0.432439, 0.413158],
            [0.0, 0.0, 0.0, 0.255314, -0.985647, 1.461641],
        ]
        self.assertAllClose(expected_x_left, real_x_left)

    def test_2D_constant_shape_trim(self):
        x = tf.random.normal(shape=(3, 3), seed=123456)
        shape = [1, 3]
        trimmed_x_right = voxel_utils._pad_or_trim_to(x, shape, pad_val=0)
        trimmed_x_left = voxel_utils._pad_or_trim_to(
            x, shape, pad_val=0, pad_after_contents=False
        )
        self.assertEqual(trimmed_x_right.shape.as_list(), [1, 3])
        self.assertEqual(trimmed_x_left.shape.as_list(), [1, 3])
        real_x_right, real_x_left = self.evaluate(
            [trimmed_x_right, trimmed_x_left]
        )
        expected_x_right = [[0.38615, 2.975221, -0.852826]]
        self.assertAllClose(expected_x_right, real_x_right)
        expected_x_left = [[0.255314, -0.985647, 1.461641]]
        self.assertAllClose(expected_x_left, real_x_left)
