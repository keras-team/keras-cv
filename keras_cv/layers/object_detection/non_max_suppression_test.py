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
import numpy as np

from keras_cv import layers
from keras_cv.backend import ops
from keras_cv.tests.test_case import TestCase


class NonMaxSupressionTest(TestCase):
    def test_confidence_threshold(self):
        boxes = np.random.uniform(low=0, high=1, size=(2, 5, 4))
        classes = ops.expand_dims(
            np.array(
                [[0.1, 0.1, 0.4, 0.9, 0.5], [0.7, 0.5, 0.3, 0.0, 0.0]],
                "float32",
            ),
            axis=-1,
        )

        nms = layers.NonMaxSuppression(
            bounding_box_format="yxyx",
            from_logits=False,
            iou_threshold=1.0,
            confidence_threshold=0.45,
            max_detections=2,
        )

        outputs = nms(boxes, classes)

        self.assertAllClose(
            outputs["boxes"], [boxes[0][-2:, ...], boxes[1][:2, ...]]
        )
        self.assertAllClose(outputs["classes"], [[0.0, 0.0], [0.0, 0.0]])
        self.assertAllClose(outputs["confidence"], [[0.9, 0.5], [0.7, 0.5]])

    def test_max_detections(self):
        boxes = np.random.uniform(low=0, high=1, size=(2, 5, 4))
        classes = ops.expand_dims(
            np.array(
                [[0.1, 0.1, 0.4, 0.5, 0.9], [0.7, 0.5, 0.3, 0.0, 0.0]],
                "float32",
            ),
            axis=-1,
        )

        nms = layers.NonMaxSuppression(
            bounding_box_format="yxyx",
            from_logits=False,
            iou_threshold=1.0,
            confidence_threshold=0.1,
            max_detections=1,
        )

        outputs = nms(boxes, classes)

        self.assertAllClose(
            outputs["boxes"], [boxes[0][-1:, ...], boxes[1][:1, ...]]
        )
        self.assertAllClose(outputs["classes"], [[0.0], [0.0]])
        self.assertAllClose(outputs["confidence"], [[0.9], [0.7]])
