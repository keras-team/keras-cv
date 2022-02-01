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
"""Tests for bbox functions."""

import tensorflow as tf

from keras_cv.metrics.coco.base import COCOBase


class COCOBaseTest(tf.test.TestCase):
    def test_area_range_bounding_box_counting(self):
        y_true = tf.constant(
            [[[0, 0, 100, 100, 1], [0, 0, 100, 100, 1]]], dtype=tf.float32
        )
        y_pred = tf.constant([[[0, 50, 100, 150, 1, 1.0]]], dtype=tf.float32)
        # note the low iou threshold
        metric = COCOBase(
            iou_thresholds=[0.15],
            category_ids=[1],
            area_range=(0, 10000**2),
            max_detections=1,
        )
        metric.update_state(y_true, y_pred)

        self.assertEqual(2.0, metric.ground_truth_boxes[0].numpy())
        self.assertEqual(1.0, metric.true_positives[0, 0].numpy())
        self.assertEqual(0.0, metric.false_positives[0, 0].numpy())

    def test_true_positive_counting_one_good_one_bad(self):
        y_true = tf.constant(
            [[[0, 0, 100, 100, 1], [0, 0, 100, 100, 1]]], dtype=tf.float32
        )
        y_pred = tf.constant([[[0, 50, 100, 150, 1, 1.0]]], dtype=tf.float32)
        # note the low iou threshold
        metric = COCOBase(
            iou_thresholds=[0.15],
            category_ids=[1],
            area_range=(0, 10000**2),
            max_detections=1,
        )
        metric.update_state(y_true, y_pred)

        self.assertEqual(2.0, metric.ground_truth_boxes[0].numpy())
        self.assertEqual(1.0, metric.true_positives[0, 0].numpy())
        self.assertEqual(0.0, metric.false_positives[0, 0].numpy())

    def test_true_positive_counting_one_true_two_pred(self):
        y_true = tf.constant(
            [
                [
                    [0, 0, 100, 100, 1],
                ]
            ],
            dtype=tf.float32,
        )
        y_pred = tf.constant(
            [[[0, 50, 100, 150, 1, 0.90], [0, 0, 100, 100, 1, 1.0]]],
            dtype=tf.float32,
        )
        # note the low iou threshold
        metric = COCOBase(
            iou_thresholds=[0.15],
            category_ids=[1],
            area_range=(0, 10000**2),
            max_detections=1,
        )
        metric.update_state(y_true, y_pred)

        self.assertEqual(1.0, metric.true_positives[0, 0].numpy())

        y_true = tf.constant([[[0, 0, 100, 100, 1]]], dtype=tf.float32)
        y_pred = tf.constant([[[0, 50, 100, 150, 1, 1.0]]], dtype=tf.float32)

        metric.update_state(y_true, y_pred)
        self.assertEqual(2.0, metric.true_positives[0, 0].numpy())

    def test_matches_single_box(self):
        y_true = tf.constant([[[0, 0, 100, 100, 1]]], dtype=tf.float32)
        y_pred = tf.constant([[[0, 50, 100, 150, 1, 1.0]]], dtype=tf.float32)

        # note the low iou threshold
        metric = COCOBase(
            iou_thresholds=[0.15],
            category_ids=[1],
            area_range=(0, 10000**2),
            max_detections=1,
        )
        metric.update_state(y_true, y_pred)

        self.assertEqual(1.0, metric.true_positives[0, 0].numpy())
        self.assertEqual(0.0, metric.false_positives[0, 0].numpy())

    def test_matches_single_false_positive(self):
        y_true = tf.constant([[[0, 0, 100, 100, 1]]], dtype=tf.float32)
        y_pred = tf.constant([[[0, 50, 100, 150, 1, 1.0]]], dtype=tf.float32)

        metric = COCOBase(
            iou_thresholds=[0.95],
            category_ids=[1],
            area_range=(0, 100000**2),
            max_detections=1,
        )
        metric.update_state(y_true, y_pred)

        self.assertEqual(0.0, metric.true_positives[0, 0].numpy())
        self.assertEqual(1.0, metric.false_positives[0, 0].numpy())

    def test_merge_state(self):
        y_true = tf.constant([[[0, 0, 100, 100, 1]]], dtype=tf.float32)
        y_pred = tf.constant([[[0, 50, 100, 150, 1, 1.0]]], dtype=tf.float32)

        m1 = COCOBase(
            iou_thresholds=[0.95],
            category_ids=[1],
            area_range=(0, 100000**2),
            max_detections=1,
        )
        m2 = COCOBase(
            iou_thresholds=[0.95],
            category_ids=[1],
            area_range=(0, 100000**2),
            max_detections=1,
        )

        m1.update_state(y_true, y_pred)
        m1.update_state(y_true, y_pred)
        m2.update_state(y_true, y_pred)

        metric_result = COCOBase(
            iou_thresholds=[0.95],
            category_ids=[1],
            area_range=(0, 100000**2),
            max_detections=1,
        )
        metric_result.merge_state([m1, m2])

        self.assertEqual(0.0, metric_result.true_positives[0, 0].numpy())
        self.assertEqual(3.0, metric_result.false_positives[0, 0].numpy())
