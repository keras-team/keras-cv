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

from keras_cv.backend import ops
from keras_cv.utils.target_gather import _target_gather


class TargetGatherTest(tf.test.TestCase):
    def test_target_gather_boxes_batched(self):
        target_boxes = np.array(
            [[0, 0, 5, 5], [0, 5, 5, 10], [5, 0, 10, 5], [5, 5, 10, 10]]
        )
        target_boxes = ops.expand_dims(target_boxes, axis=0)
        indices = np.array([[0, 2]], dtype="int32")
        expected_boxes = np.array([[0, 0, 5, 5], [5, 0, 10, 5]])
        expected_boxes = ops.expand_dims(expected_boxes, axis=0)
        res = _target_gather(target_boxes, indices)
        self.assertAllClose(expected_boxes, res)

    def test_target_gather_boxes_unbatched(self):
        target_boxes = np.array(
            [[0, 0, 5, 5], [0, 5, 5, 10], [5, 0, 10, 5], [5, 5, 10, 10]],
            "int32",
        )
        indices = np.array([0, 2], dtype="int32")
        expected_boxes = np.array([[0, 0, 5, 5], [5, 0, 10, 5]])
        res = _target_gather(target_boxes, indices)
        self.assertAllClose(expected_boxes, res)

    def test_target_gather_classes_batched(self):
        target_classes = np.array([[1, 2, 3, 4]])
        target_classes = ops.expand_dims(target_classes, axis=-1)
        indices = np.array([[0, 2]], dtype="int32")
        expected_classes = np.array([[1, 3]])
        expected_classes = ops.expand_dims(expected_classes, axis=-1)
        res = _target_gather(target_classes, indices)
        self.assertAllClose(expected_classes, res)

    def test_target_gather_classes_unbatched(self):
        target_classes = np.array([1, 2, 3, 4])
        target_classes = ops.expand_dims(target_classes, axis=-1)
        indices = np.array([0, 2], dtype="int32")
        expected_classes = np.array([1, 3])
        expected_classes = ops.expand_dims(expected_classes, axis=-1)
        res = _target_gather(target_classes, indices)
        self.assertAllClose(expected_classes, res)

    def test_target_gather_classes_batched_with_mask(self):
        target_classes = np.array([[1, 2, 3, 4]])
        target_classes = ops.expand_dims(target_classes, axis=-1)
        indices = np.array([[0, 2]], dtype="int32")
        masks = np.array(([[False, True]]))
        masks = ops.expand_dims(masks, axis=-1)
        # the second element is masked
        expected_classes = np.array([[1, 0]])
        expected_classes = ops.expand_dims(expected_classes, axis=-1)
        res = _target_gather(target_classes, indices, masks)
        self.assertAllClose(expected_classes, res)

    def test_target_gather_classes_batched_with_mask_val(self):
        target_classes = np.array([[1, 2, 3, 4]])
        target_classes = ops.expand_dims(target_classes, axis=-1)
        indices = np.array([[0, 2]], dtype="int32")
        masks = np.array(([[False, True]]))
        masks = ops.expand_dims(masks, axis=-1)
        # the second element is masked
        expected_classes = np.array([[1, -1]])
        expected_classes = ops.expand_dims(expected_classes, axis=-1)
        res = _target_gather(target_classes, indices, masks, -1)
        self.assertAllClose(expected_classes, res)

    def test_target_gather_classes_unbatched_with_mask(self):
        target_classes = np.array([1, 2, 3, 4])
        target_classes = ops.expand_dims(target_classes, axis=-1)
        indices = np.array([0, 2], dtype="int32")
        masks = np.array([False, True])
        masks = ops.expand_dims(masks, axis=-1)
        expected_classes = np.array([1, 0])
        expected_classes = ops.expand_dims(expected_classes, axis=-1)
        res = _target_gather(target_classes, indices, masks)
        self.assertAllClose(expected_classes, res)

    def test_target_gather_with_empty_targets(self):
        target_classes = np.array([])
        target_classes = ops.expand_dims(target_classes, axis=-1)
        indices = np.array([0, 2], dtype="int32")
        # return all 0s since input is empty
        expected_classes = np.array([0, 0])
        expected_classes = ops.expand_dims(expected_classes, axis=-1)
        res = _target_gather(target_classes, indices)
        self.assertAllClose(expected_classes, res)

    def test_target_gather_classes_multi_batch(self):
        target_classes = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        target_classes = ops.expand_dims(target_classes, axis=-1)
        indices = np.array([[0, 2], [1, 3]], dtype="int32")
        expected_classes = np.array([[1, 3], [6, 8]])
        expected_classes = ops.expand_dims(expected_classes, axis=-1)
        res = _target_gather(target_classes, indices)
        self.assertAllClose(expected_classes, res)

    def test_target_gather_invalid_rank(self):
        targets = np.random.normal(size=[32, 2, 2, 2])
        indices = np.array([0, 1], dtype="int32")
        with self.assertRaisesRegex(ValueError, "larger than 3"):
            _ = _target_gather(targets, indices)
