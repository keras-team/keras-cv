"""Tests for COCORecall."""

import numpy as np
import tensorflow as tf

from keras_cv.metrics.coco.recall import COCORecall


class COCORecallTest(tf.test.TestCase):
    def test_recall_direct_assignment(self):
        recall = COCORecall(max_detections=[1e9], category_ids=[1], area_ranges=[(0, 1e9**2)])
        t = recall.iou_thresholds.shape[0]
        k = recall.category_ids.shape[0]

        true_positives = tf.ones((t, k, 1, 1))
        ground_truth_boxes = true_positives[0] * 2
        recall.true_positives.assign(true_positives)
        recall.ground_truth_boxes.assign(ground_truth_boxes)

        self.assertEqual(recall.result(), 0.5)
