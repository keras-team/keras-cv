"""Tests for iou functions."""

import numpy as np
import tensorflow as tf

from keras_cv.metrics.coco import iou as iou_lib


class IoUTest(tf.test.TestCase):
    def test_compute_single_iou(self):
        bb1 = tf.constant([[100, 101, 200, 201]], dtype=tf.float32)
        bb1_off_by_1 = tf.constant([[101, 102, 201, 202]], dtype=tf.float32)
        # area of bb1 and bb1_off_by_1 are each 10000.
        # intersection area is 99*99=9801
        # iou=9801/(2*10000 - 9801)=0.96097656633
        print(iou_lib.compute_ious_for_image(bb1, bb1_off_by_1))
        self.assertAlmostEqual(
            iou_lib.compute_ious_for_image(bb1, bb1_off_by_1)[0], 0.96097656633
        )

    def test_compute_ious_for_image(self):

        bb1 = [100, 101, 200, 201]
        bb1_off_by_1_pred = [101, 102, 201, 202]
        iou_bb1_bb1_off = 0.96097656633
        top_left_bounding_box = [0, 2, 1, 3]
        far_away_box = [1300, 1400, 1500, 1401]
        another_far_away_pred = [1000, 1400, 1200, 1401]

        # Rows represent predictions, columns ground truths
        expected_result = np.array(
            [[iou_bb1_bb1_off, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )

        sample_y_true = tf.constant(
            [bb1, top_left_bounding_box, far_away_box], dtype=tf.float32
        )
        sample_y_pred = tf.constant(
            [bb1_off_by_1_pred, top_left_bounding_box, another_far_away_pred],
            dtype=tf.float32,
        )

        result = iou_lib.compute_ious_for_image(sample_y_true, sample_y_pred)
        self.assertAllClose(expected_result, result.numpy())
