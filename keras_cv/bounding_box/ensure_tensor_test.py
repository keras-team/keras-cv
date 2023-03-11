# Copyright 2023 The KerasCV Authors
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


class BoundingBoxEnsureTensorTest(tf.test.TestCase):
    def test_convert_list(self):
        boxes = {"boxes": [[0, 1, 2, 3]], "classes": [0]}
        output = bounding_box.ensure_tensor(boxes)
        self.assertFalse(
            any([isinstance(boxes[k], tf.Tensor) for k in boxes.keys()])
        )
        self.assertTrue(
            all([isinstance(output[k], tf.Tensor) for k in output.keys()])
        )

    def test_confidence(self):
        boxes = {"boxes": [[0, 1, 2, 3]], "classes": [0], "confidence": [0.245]}
        output = bounding_box.ensure_tensor(boxes)
        self.assertFalse(
            any([isinstance(boxes[k], tf.Tensor) for k in boxes.keys()])
        )
        self.assertTrue(
            all([isinstance(output[k], tf.Tensor) for k in output.keys()])
        )
