# Copyright 2024 The KerasCV Authors
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
from absl.testing import parameterized

from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.backend.config import keras_3
from keras_cv.src.models.object_detection.faster_rcnn import RPNHead
from keras_cv.src.tests.test_case import TestCase


class RCNNHeadTest(TestCase):
    @pytest.mark.skipif(not keras_3(), reason="disabling test for Keras 2")
    def test_return_type_dict(
        self,
    ):
        layer = RPNHead()
        c2 = ops.ones([2, 64, 64, 256])
        c3 = ops.ones([2, 32, 32, 256])
        c4 = ops.ones([2, 16, 16, 256])
        c5 = ops.ones([2, 8, 8, 256])
        c6 = ops.ones([2, 4, 4, 256])

        inputs = {"P2": c2, "P3": c3, "P4": c4, "P5": c5, "P6": c6}
        rpn_boxes, rpn_scores = layer(inputs)
        self.assertTrue(isinstance(rpn_boxes, dict))
        self.assertTrue(isinstance(rpn_scores, dict))
        self.assertEquals(
            sorted(rpn_boxes.keys()), ["P2", "P3", "P4", "P5", "P6"]
        )
        self.assertEquals(
            sorted(rpn_scores.keys()), ["P2", "P3", "P4", "P5", "P6"]
        )

    @pytest.mark.skipif(not keras_3(), reason="disabling test for Keras 2")
    def test_return_type_list(self):
        layer = RPNHead()
        c2 = ops.ones([2, 64, 64, 256])
        c3 = ops.ones([2, 32, 32, 256])
        c4 = ops.ones([2, 16, 16, 256])
        c5 = ops.ones([2, 8, 8, 256])
        c6 = ops.ones([2, 4, 4, 256])

        inputs = [c2, c3, c4, c5, c6]
        rpn_boxes, rpn_scores = layer(inputs)
        self.assertTrue(isinstance(rpn_boxes, list))
        self.assertTrue(isinstance(rpn_scores, list))

    @pytest.mark.skipif(not keras_3(), reason="disabling test for Keras 2")
    @parameterized.parameters(
        (3,),
        (9,),
    )
    def test_with_keras_input_tensor_and_num_anchors(self, num_anchors):
        layer = RPNHead(num_anchors_per_location=num_anchors)
        c2 = keras.layers.Input([64, 64, 256])
        c3 = keras.layers.Input([32, 32, 256])
        c4 = keras.layers.Input([16, 16, 256])
        c5 = keras.layers.Input([8, 8, 256])
        c6 = keras.layers.Input([4, 4, 256])

        inputs = {"P2": c2, "P3": c3, "P4": c4, "P5": c5, "P6": c6}
        rpn_boxes, rpn_scores = layer(inputs)
        for level in inputs.keys():
            self.assertEquals(rpn_boxes[level].shape[1], inputs[level].shape[1])
            self.assertEquals(rpn_boxes[level].shape[2], inputs[level].shape[2])
            self.assertEquals(rpn_boxes[level].shape[3], layer.num_anchors * 4)

        for level in inputs.keys():
            self.assertEquals(
                rpn_scores[level].shape[1], inputs[level].shape[1]
            )
            self.assertEquals(
                rpn_scores[level].shape[2], inputs[level].shape[2]
            )
            self.assertEquals(rpn_scores[level].shape[3], layer.num_anchors * 1)
