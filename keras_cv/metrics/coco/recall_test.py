"""Tests for COCORecall."""

import numpy as np
import tensorflow as tf

from keras_cv.metrics.coco.recall import COCORecall


class COCORecallTest(tf.test.TestCase):
    def test_recall_direct_assignment(self):
        recall = COCORecall(
            max_detections=[1e9], category_ids=[1], area_ranges=[(0, 1e9 ** 2)]
        )
        t = recall.iou_thresholds.shape[0]
        k = recall.category_ids.shape[0]

        true_positives = tf.ones((t, k, 1, 1))
        ground_truth_boxes = tf.ones((k, 1)) * 2
        recall.true_positives.assign(true_positives)
        recall.ground_truth_boxes.assign(ground_truth_boxes)

        self.assertEqual(recall.result(), 0.5)

    def test_max_detections_one_third(self):
        recall = COCORecall(
            max_detections=[1], category_ids=[1], area_ranges=[(0, 1e9 ** 2)]
        )
        y_true = np.array(
            [
                [
                    [0, 0, 100, 100, 1],
                    [100, 100, 200, 200, 1],
                    [300, 300, 400, 400, 1],
                ]
            ]
        ).astype(np.float32)
        y_pred = np.concatenate([y_true, np.ones((1, 3, 1))], axis=-1).astype(
            np.float32
        )
        # with max_dets=1, only 1 of the three boxes can be found
        recall.update_state(y_true, y_pred)

        self.assertAlmostEqual(recall.result().numpy(), 1 / 3)

    def test_max_detections(self):
        recall = COCORecall(
            max_detections=[3], category_ids=[1], area_ranges=[(0, 1e9 ** 2)]
        )
        y_true = np.array(
            [
                [
                    [0, 0, 100, 100, 1],
                    [100, 100, 200, 200, 1],
                    [300, 300, 400, 400, 1],
                ]
            ]
        ).astype(np.float32)
        y_pred = np.concatenate([y_true, np.ones((1, 3, 1))], axis=-1).astype(
            np.float32
        )

        # with max_dets=1, only 1 of the three boxes can be found
        recall.update_state(y_true, y_pred)

        self.assertAlmostEqual(recall.result().numpy(), 1.0)

    # TODO(lukewood): need to implement a test where the detections are inconsistent across thresholds
    # TODO(lukewood): need to implement a test where the detections are inconsistent across categories

    def test_recall_direct_assignment_one_third(self):
        recall = COCORecall(
            max_detections=[1e9], category_ids=[1], area_ranges=[(0, 1e9 ** 2)]
        )
        t = recall.iou_thresholds.shape[0]
        k = recall.category_ids.shape[0]

        true_positives = tf.ones((t, k, 1, 1))
        ground_truth_boxes = tf.ones((k, 1)) * 3

        recall.true_positives.assign(true_positives)
        recall.ground_truth_boxes.assign(ground_truth_boxes)

        self.assertAlmostEqual(recall.result().numpy(), 1 / 3)

    # TODO(lukewood): need to implement a test where the detections are inconsistent across thresholds
    # TODO(lukewood): need to implement a test where the detections are inconsistent across categories
