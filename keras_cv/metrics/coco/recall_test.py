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
"""Tests for COCORecall."""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras_cv.metrics import COCORecall


class COCORecallTest(tf.test.TestCase):
    def test_runs_inside_model(self):
        i = keras.layers.Input((None, None, 6))
        model = keras.Model(i, i)

        recall = COCORecall(
            max_detections=100,
            bounding_box_format="xyxy",
            class_ids=[1],
            area_range=(0, 64**2),
        )

        # These would match if they were in the area range
        y_true = np.array([[[0, 0, 10, 10, 1], [5, 5, 10, 10, 1]]]).astype(np.float32)
        y_pred = np.array([[[0, 0, 10, 10, 1, 1.0], [5, 5, 10, 10, 1, 0.9]]]).astype(
            np.float32
        )

        model.compile(metrics=[recall])
        model.evaluate(y_pred, y_true)

        self.assertAllEqual(recall.result(), 1.0)

    def test_ragged_tensor_support(self):
        recall = COCORecall(
            max_detections=100,
            bounding_box_format="xyxy",
            class_ids=[1],
            area_range=(0, 64**2),
        )

        # These would match if they were in the area range
        y_true = tf.ragged.stack(
            [
                tf.constant([[0, 0, 10, 10, 1], [5, 5, 10, 10, 1]], tf.float32),
                tf.constant([[0, 0, 10, 10, 1]], tf.float32),
            ]
        )
        y_pred = tf.ragged.stack(
            [
                tf.constant([[5, 5, 10, 10, 1, 0.9]], tf.float32),
                tf.constant(
                    [[0, 0, 10, 10, 1, 1.0], [5, 5, 10, 10, 1, 0.9]], tf.float32
                ),
            ]
        )

        recall.update_state(y_true, y_pred)
        self.assertAlmostEqual(recall.result(), 2 / 3)

    def test_merge_state(self):
        y_true = tf.constant([[[0, 0, 100, 100, 1]]], dtype=tf.float32)
        y_pred = tf.constant([[[0, 50, 100, 150, 1, 1.0]]], dtype=tf.float32)
        y_pred_match = tf.constant([[[0, 0, 100, 100, 1, 1.0]]], dtype=tf.float32)

        m1 = COCORecall(
            bounding_box_format="xyxy",
            iou_thresholds=[0.95],
            class_ids=[1],
            area_range=(0, 100000**2),
            max_detections=1,
        )
        m2 = COCORecall(
            bounding_box_format="xyxy",
            iou_thresholds=[0.95],
            class_ids=[1],
            area_range=(0, 100000**2),
            max_detections=1,
        )

        m1.update_state(y_true, y_pred)
        m1.update_state(y_true, y_pred_match)

        m2.update_state(y_true, y_pred)

        metric_result = COCORecall(
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
        recall = COCORecall(
            bounding_box_format="xyxy",
            max_detections=100,
            class_ids=[1],
            area_range=(32**2, 64**2),
        )

        # These would match if they were in the area range
        y_true = np.array([[[0, 0, 10, 10, 1], [5, 5, 10, 10, 1]]]).astype(np.float32)
        y_pred = np.array([[[0, 0, 10, 10, 1, 1.0], [5, 5, 10, 10, 1, 0.9]]]).astype(
            np.float32
        )
        recall.update_state(y_true, y_pred)

        self.assertAllEqual(recall.result(), 0.0)

    def test_missing_categories(self):
        recall = COCORecall(
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
        recall = COCORecall(
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
        recall = COCORecall(
            bounding_box_format="xyxy",
            max_detections=1,
            class_ids=[1],
            area_range=(0, 1e9**2),
        )
        y_true = np.array(
            [[[0, 0, 100, 100, 1], [100, 100, 200, 200, 1], [300, 300, 400, 400, 1]]]
        ).astype(np.float32)
        y_pred = np.concatenate([y_true, np.ones((1, 3, 1))], axis=-1).astype(
            np.float32
        )
        # with max_dets=1, only 1 of the three boxes can be found
        recall.update_state(y_true, y_pred)

        self.assertAlmostEqual(recall.result().numpy(), 1 / 3)

    def test_max_detections(self):
        recall = COCORecall(
            bounding_box_format="xyxy",
            max_detections=3,
            class_ids=[1],
            area_range=(0, 1e9**2),
        )
        y_true = np.array(
            [[[0, 0, 100, 100, 1], [100, 100, 200, 200, 1], [300, 300, 400, 400, 1]]]
        ).astype(np.float32)
        y_pred = np.concatenate([y_true, np.ones((1, 3, 1))], axis=-1).astype(
            np.float32
        )

        # with max_dets=1, only 1 of the three boxes can be found
        recall.update_state(y_true, y_pred)

        self.assertAlmostEqual(recall.result().numpy(), 1.0)

    def test_recall_direct_assignment_one_third(self):
        recall = COCORecall(
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
        y_true = tf.constant(
            [[[0, 0, 100, 100, 1], [0, 0, 100, 100, 1]]], dtype=tf.float32
        )
        y_pred = tf.constant([[[0, 50, 100, 150, 1, 1.0]]], dtype=tf.float32)
        # note the low iou threshold
        metric = COCORecall(
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
        y_true = tf.constant(
            [[[0, 0, 100, 100, 1], [0, 0, 100, 100, 1]]], dtype=tf.float32
        )
        y_pred = tf.constant([[[0, 50, 100, 150, 1, 1.0]]], dtype=tf.float32)
        # note the low iou threshold
        metric = COCORecall(
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
        y_true = tf.constant(
            [[[0, 0, 100, 100, 1]]],
            dtype=tf.float32,
        )
        y_pred = tf.constant(
            [[[0, 50, 100, 150, 1, 0.90], [0, 0, 100, 100, 1, 1.0]]],
            dtype=tf.float32,
        )
        # note the low iou threshold
        metric = COCORecall(
            bounding_box_format="xyxy",
            iou_thresholds=[0.15],
            class_ids=[1],
            area_range=(0, 10000**2),
            max_detections=1,
        )
        metric.update_state(y_true, y_pred)
        self.assertEqual([[1]], metric.true_positives)

        y_true = tf.constant([[[0, 0, 100, 100, 1]]], dtype=tf.float32)
        y_pred = tf.constant([[[0, 50, 100, 150, 1, 1.0]]], dtype=tf.float32)

        metric.update_state(y_true, y_pred)
        self.assertEqual([[2]], metric.true_positives)

    def test_mixed_dtypes(self):
        y_true = tf.constant([[[0, 0, 100, 100, 1]]], dtype=tf.float64)
        y_pred = tf.constant([[[0, 50, 100, 150, 1, 1.0]]], dtype=tf.float32)

        metric = COCORecall(
            bounding_box_format="xyxy",
            iou_thresholds=[0.15],
            class_ids=[1],
            area_range=(0, 10000**2),
            max_detections=1,
        )
        metric.update_state(y_true, y_pred)
        self.assertEqual(metric.result(), 1.0)

    def test_matches_single_box(self):
        y_true = tf.constant([[[0, 0, 100, 100, 1]]], dtype=tf.float32)
        y_pred = tf.constant([[[0, 50, 100, 150, 1, 1.0]]], dtype=tf.float32)

        # note the low iou threshold
        metric = COCORecall(
            bounding_box_format="xyxy",
            iou_thresholds=[0.15],
            class_ids=[1],
            area_range=(0, 10000**2),
            max_detections=1,
        )
        metric.update_state(y_true, y_pred)

        self.assertEqual([[1]], metric.true_positives)

    def test_matches_single_false_positive(self):
        y_true = tf.constant([[[0, 0, 100, 100, 1]]], dtype=tf.float32)
        y_pred = tf.constant([[[0, 50, 100, 150, 1, 1.0]]], dtype=tf.float32)

        metric = COCORecall(
            bounding_box_format="xyxy",
            iou_thresholds=[0.95],
            class_ids=[1],
            area_range=(0, 100000**2),
            max_detections=1,
        )
        metric.update_state(y_true, y_pred)

        self.assertEqual([[0]], metric.true_positives)
