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

import keras_cv


class SmoothL1LossTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("none", "none", (20,)),
        ("sum", "sum", ()),
        ("sum_over_batch_size", "sum_over_batch_size", ()),
    )
    def test_proper_output_shapes(self, reduction, target_size):
        loss = keras_cv.losses.SmoothL1Loss(l1_cutoff=0.5, reduction=reduction)
        result = loss(
            y_true=tf.random.uniform((20, 300)),
            y_pred=tf.random.uniform((20, 300)),
        )
        self.assertEqual(result.shape, target_size)
