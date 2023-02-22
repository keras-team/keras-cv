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
"""Tests for _BoxRecall."""

import numpy as np
import tensorflow as tf

from keras_cv import bounding_box
from keras_cv.metrics import _BoxRecall


class BoxRecallTest(tf.test.TestCase):
    def test_ragged_tensor_support(self):
        recall = _BoxRecall(
            max_detections=100,
            bounding_box_format="xyxy",
            class_ids=[1],
            area_range=(0, 64**2),
        )

        # These would match if they were in the area range
        y_true = {
            "boxes": tf.ragged.stack(
                [
                    tf.constant([[0, 0, 10, 10], [5, 5, 10, 10]], tf.float32),
                    tf.constant([[0, 0, 10, 10]], tf.float32),
                ]
            ),
            "classes": tf.ragged.stack([tf.constant([1, 1]), tf.constant([1])]),
        }

        y_pred = {
            "boxes": tf.ragged.stack(
                [
                    tf.constant([[5, 5, 10, 10]], tf.float32),
                    tf.constant([[0, 0, 10, 10], [5, 5, 10, 10]], tf.float32),
                ]
            ),
            "classes": tf.ragged.stack([tf.constant([1]), tf.constant([1, 1])]),
            "confidence": tf.ragged.stack(
                [tf.constant([1.0]), tf.constant([1.0, 0.9])]
            ),
        }

        recall.update_state(y_true, y_pred)
        self.assertAlmostEqual(recall.result(), 2 / 3)

    def test_merge_state(self):
        y_true = {
            "boxes": [[[0, 0, 100, 100]]],
            "classes": [[1]],
        }
        y_pred = {
            "boxes": [[[0, 50, 100, 150]]],
            "classes": [[1]],
            "confidence": [[1.0]],
        }
        y_pred_match = {
            "boxes": [[[0, 0, 100, 100]]],
            "classes": [[1]],
            "confidence": [[1.0]],
        }
        m1 = _BoxRecall(
            bounding_box_format="xyxy",
            iou_thresholds=[0.95],
            class_ids=[1],
            area_range=(0, 100000**2),
            max_detections=1,
        )
        m2 = _BoxRecall(
            bounding_box_format="xyxy",
            iou_thresholds=[0.95],
            class_ids=[1],
            area_range=(0, 100000**2),
            max_detections=1,
        )

        m1.update_state(y_true, y_pred)
        m1.update_state(y_true, y_pred_match)

        m2.update_state(y_true, y_pred)

        metric_result = _BoxRecall(
            bounding_box_format="xyxy",
            iou_thresholds=[0.95],
            class_ids=[1],
            area_range=(0, 100000**2),
            max_detections=1,
        )
        metric_result.merge_state([m1, m2])

        self.assertEqual([[1]], metric_result.true_positives)
        self.assertEqual([3], metric_result.ground_truth_boxes)
        self.assertEqual(1 / 3, metric_result.result())

    def test_recall_area_range_filtering(self):
        recall = _BoxRecall(
            bounding_box_format="xyxy",
            max_detections=100,
            class_ids=[1],
            area_range=(32**2, 64**2),
        )
        y_true = {
            "boxes": [[[0, 0, 10, 10], [5, 5, 10, 10]]],
            "classes": [[1, 1]],
        }
        y_pred = {
            "boxes": [[[0, 0, 10, 10], [5, 5, 10, 10]]],
            "classes": [[1, 1]],
            "confidence": [[1.0, 0.9]],
        }

        recall.update_state(y_true, y_pred)

        self.assertAllEqual(recall.result(), 0.0)

    def test_missing_categories(self):
        recall = _BoxRecall(
            bounding_box_format="xyxy",
            max_detections=100,
            class_ids=[1, 2, 3],
            area_range=(0, 1e9**2),
        )
        t = len(recall.iou_thresholds)
        k = len(recall.class_ids)

        true_positives = np.ones((t, k))
        true_positives[:, 1] = np.zeros((t,))
        true_positives = tf.constant(true_positives, dtype=tf.int32)

        ground_truth_boxes = np.ones((k,)) * 2
        ground_truth_boxes[1] = 0

        ground_truth_boxes = tf.constant(ground_truth_boxes, dtype=tf.int32)
        recall.true_positives.assign(true_positives)
        recall.ground_truth_boxes.assign(ground_truth_boxes)

        self.assertEqual(recall.result(), 0.5)

    def test_recall_direct_assignment(self):
        recall = _BoxRecall(
            bounding_box_format="xyxy",
            max_detections=100,
            class_ids=[1],
            area_range=(0, 1e9**2),
        )
        t = len(recall.iou_thresholds)
        k = len(recall.class_ids)

        true_positives = tf.ones((t, k), dtype=tf.int32)
        ground_truth_boxes = tf.ones((k,), dtype=tf.int32) * 2
        recall.true_positives.assign(true_positives)
        recall.ground_truth_boxes.assign(ground_truth_boxes)

        self.assertEqual(recall.result(), 0.5)

    def test_max_detections_one_third(self):
        recall = _BoxRecall(
            bounding_box_format="xyxy",
            max_detections=1,
            class_ids=[1],
            area_range=(0, 1e9**2),
        )
        y_true = {
            "boxes": [
                [
                    [0, 0, 100, 100],
                    [100, 100, 200, 200],
                    [300, 300, 400, 400],
                ]
            ],
            "classes": [[1, 1, 1]],
        }
        y_pred = {
            "boxes": [
                [
                    [0, 0, 100, 100],
                    [100, 100, 200, 200],
                    [300, 300, 400, 400],
                ]
            ],
            "classes": [[1, 1, 1]],
            "confidence": [[1, 1, 1]],
        }
        # with max_dets=1, only 1 of the three boxes can be found
        recall.update_state(y_true, y_pred)

        self.assertAlmostEqual(recall.result().numpy(), 1 / 3)

    def test_max_detections(self):
        recall = _BoxRecall(
            bounding_box_format="xyxy",
            max_detections=3,
            class_ids=[1],
            area_range=(0, 1e9**2),
        )
        y_true = {
            "boxes": [
                [
                    [0, 0, 100, 100],
                    [100, 100, 200, 200],
                    [300, 300, 400, 400],
                ]
            ],
            "classes": [[1, 1, 1]],
        }
        y_pred = {
            "boxes": [
                [
                    [0, 0, 100, 100],
                    [100, 100, 200, 200],
                    [300, 300, 400, 400],
                ]
            ],
            "classes": [[1, 1, 1]],
            "confidence": [[1, 1, 1]],
        }

        # with max_dets=1, only 1 of the three boxes can be found
        recall.update_state(y_true, y_pred)

        self.assertAlmostEqual(recall.result().numpy(), 1.0)

    def test_recall_direct_assignment_one_third(self):
        recall = _BoxRecall(
            bounding_box_format="xyxy",
            max_detections=100,
            class_ids=[1],
            area_range=(0, 1e9**2),
        )
        t = len(recall.iou_thresholds)
        k = len(recall.class_ids)

        true_positives = tf.ones((t, k), dtype=tf.int32)
        ground_truth_boxes = tf.ones((k,), dtype=tf.int32) * 3

        recall.true_positives.assign(true_positives)
        recall.ground_truth_boxes.assign(ground_truth_boxes)

        self.assertAlmostEqual(recall.result().numpy(), 1 / 3)

    def test_area_range_bounding_box_counting(self):
        y_true = {
            "boxes": [[[0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 100.0, 100.0]]],
            "classes": [[1.0, 1.0]],
        }
        y_pred = {
            "boxes": [[[0.0, 50.0, 100.0, 150.0]]],
            "classes": [[1.0]],
            "confidence": [[1.0]],
        }
        # note the low iou threshold
        metric = _BoxRecall(
            bounding_box_format="xyxy",
            iou_thresholds=[0.15],
            class_ids=[1],
            area_range=(0, 10000**2),
            max_detections=1,
        )
        metric.update_state(y_true, y_pred)
        self.assertEqual([[2]], metric.ground_truth_boxes)
        self.assertEqual([[1]], metric.true_positives)

    def test_true_positive_counting_one_good_one_bad(self):
        y_true = {
            "boxes": [[[0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 100.0, 100.0]]],
            "classes": [[1.0, 1.0]],
        }
        y_pred = {
            "boxes": [[[0.0, 50.0, 100.0, 150.0]]],
            "classes": [[1.0]],
            "confidence": [[1.0]],
        }
        # note the low iou threshold
        metric = _BoxRecall(
            bounding_box_format="xyxy",
            iou_thresholds=[0.15],
            class_ids=[1],
            area_range=(0, 10000**2),
            max_detections=1,
        )
        metric.update_state(y_true, y_pred)

        self.assertEqual([2], metric.ground_truth_boxes)
        self.assertEqual([[1]], metric.true_positives)

    def test_true_positive_counting_one_true_two_pred(self):
        y_true = {"boxes": [[[0.0, 0.0, 100.0, 100.0]]], "classes": [[1.0]]}
        y_pred = {
            "boxes": [[[0.0, 50.0, 100.0, 150.0], [0.0, 0.0, 100.0, 100.0]]],
            "classes": [[1.0, 1.0]],
            "confidence": [[0.8999999761581421, 1.0]],
        }
        # note the low iou threshold
        metric = _BoxRecall(
            bounding_box_format="xyxy",
            iou_thresholds=[0.15],
            class_ids=[1],
            area_range=(0, 10000**2),
            max_detections=1,
        )
        metric.update_state(y_true, y_pred)
        self.assertEqual([[1]], metric.true_positives)

        y_true = {"boxes": [[[0.0, 0.0, 100.0, 100.0]]], "classes": [[1.0]]}
        y_pred = {
            "boxes": [[[0.0, 50.0, 100.0, 150.0]]],
            "classes": [[1.0]],
            "confidence": [[1.0]],
        }

        metric.update_state(y_true, y_pred)
        self.assertEqual([[2]], metric.true_positives)

    def test_mixed_dtypes(self):
        y_true = {"boxes": [[[0.0, 0.0, 100.0, 100.0]]], "classes": [[1.0]]}
        y_true = bounding_box.ensure_tensor(y_true, dtype=tf.float64)
        y_pred = {
            "boxes": [[[0.0, 50.0, 100.0, 150.0]]],
            "classes": [[1.0]],
            "confidence": [[1.0]],
        }

        metric = _BoxRecall(
            bounding_box_format="xyxy",
            iou_thresholds=[0.15],
            class_ids=[1],
            area_range=(0, 10000**2),
            max_detections=1,
        )
        metric.update_state(y_true, y_pred)
        self.assertEqual(metric.result(), 1.0)

    def test_matches_single_box(self):
        y_true = {"boxes": [[[0.0, 0.0, 100.0, 100.0]]], "classes": [[1.0]]}
        y_pred = {
            "boxes": [[[0.0, 50.0, 100.0, 150.0]]],
            "classes": [[1.0]],
            "confidence": [[1.0]],
        }

        # note the low iou threshold
        metric = _BoxRecall(
            bounding_box_format="xyxy",
            iou_thresholds=[0.15],
            class_ids=[1],
            area_range=(0, 10000**2),
            max_detections=1,
        )
        metric.update_state(y_true, y_pred)

        self.assertEqual([[1]], metric.true_positives)

    def test_matches_single_false_positive(self):
        y_true = {"boxes": [[[0.0, 0.0, 100.0, 100.0]]], "classes": [[1.0]]}
        y_pred = {
            "boxes": [[[0.0, 50.0, 100.0, 150.0]]],
            "classes": [[1.0]],
            "confidence": [[1.0]],
        }

        metric = _BoxRecall(
            bounding_box_format="xyxy",
            iou_thresholds=[0.95],
            class_ids=[1],
            area_range=(0, 100000**2),
            max_detections=1,
        )
        metric.update_state(y_true, y_pred)

        self.assertEqual([[0]], metric.true_positives)
