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

from keras_cv import bounding_box
from keras_cv.tests.test_case import TestCase


class ToDenseTest(TestCase):
    @pytest.mark.tf_keras_only
    def test_converts_to_dense(self):
        bounding_boxes = {
            "boxes": tf.ragged.constant(
                [[[0, 0, 1, 1]], [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]]
            ),
            "classes": tf.ragged.constant([[0], [1, 2, 3]]),
        }
        bounding_boxes = bounding_box.to_dense(bounding_boxes)
        self.assertEqual(bounding_boxes["boxes"].shape, [2, 3, 4])
        self.assertEqual(bounding_boxes["classes"].shape, [2, 3])
