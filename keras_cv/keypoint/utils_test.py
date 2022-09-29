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
from absl.testing import parameterized

from keras_cv.keypoint.utils import filter_out_of_image


class UtilsTestCase(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        (
            "all inside",
            tf.constant([[10.0, 20.0], [30.0, 40.0], [50.0, 50.0]]),
            tf.zeros([100, 100, 3]),
            tf.ragged.constant([[10.0, 20.0], [30.0, 40.0], [50.0, 50.0]]),
        ),
        (
            "some inside",
            tf.constant([[10.0, 20.0], [30.0, 40.0], [50.0, 50.0]]),
            tf.zeros([50, 50, 3]),
            tf.ragged.constant([[10.0, 20.0], [30.0, 40.0]]),
        ),
        (
            "ragged input",
            tf.RaggedTensor.from_row_lengths(
                [[10.0, 20.0], [30.0, 40.0], [50.0, 50.0]], [2, 1]
            ),
            tf.zeros([50, 50, 3]),
            tf.RaggedTensor.from_row_lengths([[10.0, 20.0], [30.0, 40.0]], [2, 0]),
        ),
        (
            "height - width confusion",
            tf.constant([[[10.0, 20.0]], [[40.0, 30.0]], [[30.0, 40.0]]]),
            tf.zeros((50, 40, 3)),
            tf.ragged.constant([[[10.0, 20.0]], [], [[30.0, 40.0]]], ragged_rank=1),
        ),
    )
    def test_result(self, keypoints, image, expected):
        self.assertAllClose(filter_out_of_image(keypoints, image), expected)
