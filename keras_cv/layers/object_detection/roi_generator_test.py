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

from keras_cv.layers.object_detection.roi_generator import ROIGenerator


class ROIGeneratorTest(tf.test.TestCase):
    def test_single_tensor(self):
        roi_generator = ROIGenerator("xyxy", nms_iou_threshold_train=0.96)
        rpn_boxes = tf.constant(
            [
                [[0, 0, 10, 10], [0.1, 0.1, 9.9, 9.9], [5, 5, 10, 10], [2, 2, 8, 8]],
            ]
        )
        expected_rois = tf.gather(rpn_boxes, [[1, 3, 2]], batch_dims=1)
        expected_rois = tf.concat([expected_rois, tf.zeros([1, 1, 4])], axis=1)
        rpn_scores = tf.constant(
            [
                [0.6, 0.9, 0.2, 0.3],
            ]
        )
        # selecting the 1st, then 3rd, then 2nd as they don't overlap
        # 0th box overlaps with 1st box
        expected_roi_scores = tf.gather(rpn_scores, [[1, 3, 2]], batch_dims=1)
        expected_roi_scores = tf.concat([expected_roi_scores, tf.zeros([1, 1])], axis=1)
        rois, roi_scores = roi_generator(rpn_boxes, rpn_scores, training=True)
        self.assertAllClose(expected_rois, rois)
        self.assertAllClose(expected_roi_scores, roi_scores)

    def test_single_level_single_batch_roi_ignore_box(self):
        roi_generator = ROIGenerator("xyxy", nms_iou_threshold_train=0.96)
        rpn_boxes = tf.constant(
            [
                [[0, 0, 10, 10], [0.1, 0.1, 9.9, 9.9], [5, 5, 10, 10], [2, 2, 8, 8]],
            ]
        )
        expected_rois = tf.gather(rpn_boxes, [[1, 3, 2]], batch_dims=1)
        expected_rois = tf.concat([expected_rois, tf.zeros([1, 1, 4])], axis=1)
        rpn_boxes = {2: rpn_boxes}
        rpn_scores = tf.constant(
            [
                [0.6, 0.9, 0.2, 0.3],
            ]
        )
        # selecting the 1st, then 3rd, then 2nd as they don't overlap
        # 0th box overlaps with 1st box
        expected_roi_scores = tf.gather(rpn_scores, [[1, 3, 2]], batch_dims=1)
        expected_roi_scores = tf.concat([expected_roi_scores, tf.zeros([1, 1])], axis=1)
        rpn_scores = {2: rpn_scores}
        rois, roi_scores = roi_generator(rpn_boxes, rpn_scores, training=True)
        self.assertAllClose(expected_rois, rois)
        self.assertAllClose(expected_roi_scores, roi_scores)

    def test_single_level_single_batch_roi_all_box(self):
        # for iou between 1st and 2nd box is 0.9604, so setting to 0.97 to
        # such that NMS would treat them as different ROIs
        roi_generator = ROIGenerator("xyxy", nms_iou_threshold_train=0.97)
        rpn_boxes = tf.constant(
            [
                [[0, 0, 10, 10], [0.1, 0.1, 9.9, 9.9], [5, 5, 10, 10], [2, 2, 8, 8]],
            ]
        )
        expected_rois = tf.gather(rpn_boxes, [[1, 0, 3, 2]], batch_dims=1)
        rpn_boxes = {2: rpn_boxes}
        rpn_scores = tf.constant(
            [
                [0.6, 0.9, 0.2, 0.3],
            ]
        )
        # selecting the 1st, then 0th, then 3rd, then 2nd as they don't overlap
        expected_roi_scores = tf.gather(rpn_scores, [[1, 0, 3, 2]], batch_dims=1)
        rpn_scores = {2: rpn_scores}
        rois, roi_scores = roi_generator(rpn_boxes, rpn_scores, training=True)
        self.assertAllClose(expected_rois, rois)
        self.assertAllClose(expected_roi_scores, roi_scores)

    def test_single_level_propose_rois(self):
        roi_generator = ROIGenerator("xyxy")
        rpn_boxes = tf.constant(
            [
                [[0, 0, 10, 10], [0.1, 0.1, 9.9, 9.9], [5, 5, 10, 10], [2, 2, 8, 8]],
                [[2, 2, 4, 4], [3, 3, 6, 6], [3.1, 3.1, 6.1, 6.1], [1, 1, 8, 8]],
            ]
        )
        expected_rois = tf.gather(rpn_boxes, [[1, 3, 2], [1, 3, 0]], batch_dims=1)
        expected_rois = tf.concat([expected_rois, tf.zeros([2, 1, 4])], axis=1)
        rpn_boxes = {2: rpn_boxes}
        rpn_scores = tf.constant([[0.6, 0.9, 0.2, 0.3], [0.1, 0.8, 0.3, 0.5]])
        # 1st batch -- selecting the 1st, then 3rd, then 2nd as they don't overlap
        # 2nd batch -- selecting the 1st, then 3rd, then 0th as they don't overlap
        expected_roi_scores = tf.gather(
            rpn_scores, [[1, 3, 2], [1, 3, 0]], batch_dims=1
        )
        expected_roi_scores = tf.concat([expected_roi_scores, tf.zeros([2, 1])], axis=1)
        rpn_scores = {2: rpn_scores}
        rois, roi_scores = roi_generator(rpn_boxes, rpn_scores, training=True)
        self.assertAllClose(expected_rois, rois)
        self.assertAllClose(expected_roi_scores, roi_scores)

    def test_two_level_single_batch_propose_rois_ignore_box(self):
        roi_generator = ROIGenerator("xyxy")
        rpn_boxes = tf.constant(
            [
                [[0, 0, 10, 10], [0.1, 0.1, 9.9, 9.9], [5, 5, 10, 10], [2, 2, 8, 8]],
                [[2, 2, 4, 4], [3, 3, 6, 6], [3.1, 3.1, 6.1, 6.1], [1, 1, 8, 8]],
            ]
        )
        expected_rois = tf.constant(
            [
                [
                    [0.1, 0.1, 9.9, 9.9],
                    [3, 3, 6, 6],
                    [1, 1, 8, 8],
                    [2, 2, 8, 8],
                    [5, 5, 10, 10],
                    [2, 2, 4, 4],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            ]
        )
        rpn_boxes = {2: rpn_boxes[0:1], 3: rpn_boxes[1:2]}
        rpn_scores = tf.constant([[0.6, 0.9, 0.2, 0.3], [0.1, 0.8, 0.3, 0.5]])
        # 1st batch -- selecting the 1st, then 3rd, then 2nd as they don't overlap
        # 2nd batch -- selecting the 1st, then 3rd, then 0th as they don't overlap
        expected_roi_scores = [
            [
                0.9,
                0.8,
                0.5,
                0.3,
                0.2,
                0.1,
                0.0,
                0.0,
            ]
        ]
        rpn_scores = {2: rpn_scores[0:1], 3: rpn_scores[1:2]}
        rois, roi_scores = roi_generator(rpn_boxes, rpn_scores, training=True)
        self.assertAllClose(expected_rois, rois)
        self.assertAllClose(expected_roi_scores, roi_scores)

    def test_two_level_single_batch_propose_rois_all_box(self):
        roi_generator = ROIGenerator("xyxy", nms_iou_threshold_train=0.99)
        rpn_boxes = tf.constant(
            [
                [[0, 0, 10, 10], [0.1, 0.1, 9.9, 9.9], [5, 5, 10, 10], [2, 2, 8, 8]],
                [[2, 2, 4, 4], [3, 3, 6, 6], [3.1, 3.1, 6.1, 6.1], [1, 1, 8, 8]],
            ]
        )
        expected_rois = tf.constant(
            [
                [
                    [0.1, 0.1, 9.9, 9.9],
                    [3, 3, 6, 6],
                    [0, 0, 10, 10],
                    [1, 1, 8, 8],
                    [2, 2, 8, 8],
                    [3.1, 3.1, 6.1, 6.1],
                    [5, 5, 10, 10],
                    [2, 2, 4, 4],
                ]
            ]
        )
        rpn_boxes = {2: rpn_boxes[0:1], 3: rpn_boxes[1:2]}
        rpn_scores = tf.constant([[0.6, 0.9, 0.2, 0.3], [0.1, 0.8, 0.3, 0.5]])
        # 1st batch -- selecting the 1st, then 0th, then 3rd, then 2nd as they don't overlap
        # 2nd batch -- selecting the 1st, then 3rd, then 2nd, then 0th as they don't overlap
        expected_roi_scores = [
            [
                0.9,
                0.8,
                0.6,
                0.5,
                0.3,
                0.3,
                0.2,
                0.1,
            ]
        ]
        rpn_scores = {2: rpn_scores[0:1], 3: rpn_scores[1:2]}
        rois, roi_scores = roi_generator(rpn_boxes, rpn_scores, training=True)
        self.assertAllClose(expected_rois, rois)
        self.assertAllClose(expected_roi_scores, roi_scores)
