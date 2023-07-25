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


from keras_cv import bounding_box
from keras_cv.backend import ops
from keras_cv.tests.test_case import TestCase


class BoundingBoxEnsureTensorTest(TestCase):
    def test_convert_list(self):
        boxes = {"boxes": [[0, 1, 2, 3]], "classes": [0]}
        output = bounding_box.ensure_tensor(boxes)
        self.assertFalse(any([ops.is_tensor(boxes[k]) for k in boxes.keys()]))
        self.assertTrue(all([ops.is_tensor(output[k]) for k in output.keys()]))

    def test_confidence(self):
        boxes = {"boxes": [[0, 1, 2, 3]], "classes": [0], "confidence": [0.245]}
        output = bounding_box.ensure_tensor(boxes)
        self.assertFalse(any([ops.is_tensor(boxes[k]) for k in boxes.keys()]))
        self.assertTrue(all([ops.is_tensor(output[k]) for k in output.keys()]))
