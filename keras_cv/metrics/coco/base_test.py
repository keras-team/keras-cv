"""Tests for bbox functions."""

import numpy as np
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
            area_range=(0, 10000 ** 2),
            max_detections=1,
        )
        metric.update_state(y_true, y_pred)
        # shape = [1, 1, 1, 1]
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
            area_range=(0, 10000 ** 2),
            max_detections=1,
        )
        metric.update_state(y_true, y_pred)
        # shape = [1, 1, 1, 1]
        self.assertEqual(2.0, metric.ground_truth_boxes[0].numpy())
        self.assertEqual(1.0, metric.true_positives[0, 0].numpy())
        self.assertEqual(0.0, metric.false_positives[0, 0].numpy())

    def test_true_positive_counting_one_true_two_pred(self):
        y_true = tf.constant([[[0, 0, 100, 100, 1],]], dtype=tf.float32)
        y_pred = tf.constant(
            [[[0, 50, 100, 150, 1, 0.90], [0, 0, 100, 100, 1, 1.0]]], dtype=tf.float32
        )
        # note the low iou threshold
        metric = COCOBase(
            iou_thresholds=[0.15],
            category_ids=[1],
            area_range=(0, 10000 ** 2),
            max_detections=1,
        )
        metric.update_state(y_true, y_pred)
        # shape = [1, 1, 1, 1]
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
            area_range=(0, 10000 ** 2),
            max_detections=1,
        )
        metric.update_state(y_true, y_pred)

        # shape = [1, 1, 1, 1]
        self.assertEqual(1.0, metric.true_positives[0, 0].numpy())
        # shape = [1, 1, 1, 1]
        self.assertEqual(0.0, metric.false_positives[0, 0].numpy())

    def test_matches_single_false_positive(self):
        y_true = tf.constant([[[0, 0, 100, 100, 1]]], dtype=tf.float32)
        y_pred = tf.constant([[[0, 50, 100, 150, 1, 1.0]]], dtype=tf.float32)

        metric = COCOBase(
            iou_thresholds=[0.95],
            category_ids=[1],
            area_range=(0, 100000 ** 2),
            max_detections=1,
        )
        metric.update_state(y_true, y_pred)

        # shape = [1, 1, 1, 1]
        self.assertEqual(0.0, metric.true_positives[0, 0].numpy())
        # shape = [1, 1, 1, 1]
        self.assertEqual(1.0, metric.false_positives[0, 0].numpy())
