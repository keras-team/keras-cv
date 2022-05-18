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

from keras_cv import bounding_box


class PadBatchToShapeTestCase(tf.test.TestCase):
    def test_bounding_box_padding(self):
        bounding_boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]
        target_shape = [3, 4]
        result = bounding_box.pad_batch_to_shape(bounding_boxes, target_shape)
        self.assertAllClose(result, [[1, 2, 3, 4], [5, 6, 7, 8], [-1, -1, -1, -1]])

        target_shape = [2, 5]
        result = bounding_box.pad_batch_to_shape(bounding_boxes, target_shape)
        self.assertAllClose(result, [[1, 2, 3, 4, -1], [5, 6, 7, 8, -1]])

        # Make sure to raise error if the rank is different between bounding_box and
        # target shape
        with self.assertRaisesRegex(ValueError, "Target shape should have same rank"):
            bounding_box.pad_batch_to_shape(bounding_boxes, [1, 2, 3])

        # Make sure raise error if the target shape is smaller
        target_shape = [3, 2]
        with self.assertRaisesRegex(
            ValueError, "Target shape should be larger than bounding box shape"
        ):
            bounding_box.pad_batch_to_shape(bounding_boxes, target_shape)
