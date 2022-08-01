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

import pytest
import tensorflow as tf
import keras_cv


class RetinaNetTest(tf.test.TestCase):
    def test_requires_proper_bounding_box_shapes(self):
        loss = keras_cv.applications.RetinaNetLoss(num_classes=20, reduction='none')

        with self.assertRaisesRegex(ValueError, 'y_true should have shape'):
            loss(y_true=tf.random.uniform((20, 300, 24)), y_pred=tf.random.uniform((20, 300, 24)))

        with self.assertRaisesRegex(ValueError, 'y_pred should have shape'):
            loss(y_true=tf.random.uniform((20, 300, 5)), y_pred=tf.random.uniform((20, 300, 6)))

        result = loss(y_true=tf.random.uniform((20, 300, 5)), y_pred=tf.random.uniform((20, 300, 24)))
        self.assertEqual(result.shape, [20,])
