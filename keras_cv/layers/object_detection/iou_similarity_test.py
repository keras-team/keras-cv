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

from keras_cv.layers.object_detection.iou_similarity import IouSimilarity


class IOUSimilarityTest(tf.test.TestCase):
    def test_iou_no_mask(self):
        sim_cal = IouSimilarity(box1_format="xyxy", box2_format="xyxy", box1_mask=True)
        boxes_1 = tf.constant(
            [
                [[0, 0, 10, 10], [0.1, 0.1, 9.9, 9.9]],
                [[5, 5, 10, 10], [2, 2, 8, 8]],
            ]
        )
        boxes_2 = tf.constant([[[1.0, 1.0, 2.0, 2.0]], [[-1.0, -1.0, -1.0, -1.0]]])
        iou = sim_cal(boxes_1, boxes_2)
        self.assertAllClose(iou[1, ...], tf.zeros([2, 1], dtype=iou.dtype))

    def test_iou_mask_box2(self):
        sim_cal = IouSimilarity(
            box1_format="xyxy", box2_format="xyxy", box1_mask=True, box2_mask=True
        )
        boxes_1 = tf.constant(
            [
                [[0, 0, 10, 10], [0.1, 0.1, 9.9, 9.9]],
                [[5, 5, 10, 10], [2, 2, 8, 8]],
            ]
        )
        boxes_2 = tf.constant([[[1.0, 1.0, 2.0, 2.0]], [[-1.0, -1.0, -1.0, -1.0]]])
        iou = sim_cal(boxes_1, boxes_2)
        self.assertAllClose(iou[1, ...], -tf.ones([2, 1], dtype=iou.dtype))

    def test_iou_mask_box1(self):
        sim_cal = IouSimilarity(box1_format="xyxy", box2_format="xyxy", box1_mask=True)
        boxes_1 = tf.constant([[[1.0, 1.0, 2.0, 2.0]], [[-1.0, -1.0, -1.0, -1.0]]])
        boxes_2 = tf.constant(
            [
                [[0, 0, 10, 10], [0.1, 0.1, 9.9, 9.9]],
                [[5, 5, 10, 10], [2, 2, 8, 8]],
            ]
        )
        iou = sim_cal(boxes_1, boxes_2)
        self.assertAllClose(iou[1, ...], -tf.ones([1, 2], dtype=iou.dtype))

    def test_iou_mask_both(self):
        sim_cal = IouSimilarity(
            box1_format="xyxy", box2_format="xyxy", box1_mask=True, box2_mask=True
        )
        boxes_1 = tf.constant([[[1.0, 1.0, 2.0, 2.0]], [[-1.0, -1.0, -1.0, -1.0]]])
        boxes_2 = tf.constant(
            [
                [[-1.0, -1.0, -1.0, -1.0], [0.1, 0.1, 9.9, 9.9]],
                [[5, 5, 10, 10], [2, 2, 8, 8]],
            ]
        )
        iou = sim_cal(boxes_1, boxes_2)
        self.assertAllClose(iou[1, ...], -tf.ones([1, 2], dtype=iou.dtype))
        self.assertAllClose(iou[0, 0, 0], -1.0)

    def test_iou_unbatched(self):
        sim_cal = IouSimilarity(
            box1_format="xyxy", box2_format="xyxy", box1_mask=True, box2_mask=True
        )
        boxes_1 = tf.constant([[-1.0, -1.0, -1.0, -1.0]])
        boxes_2 = tf.constant([[-1.0, -1.0, -1.0, -1.0], [0.1, 0.1, 9.9, 9.9]])
        iou = sim_cal(boxes_1, boxes_2)
        self.assertAllClose(iou, -tf.ones([1, 2], dtype=iou.dtype))
