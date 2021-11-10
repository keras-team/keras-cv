"""Tests for COCORecall."""

import tensorflow as tf
from google3.testing.pybase import googletest

from keras_cv.metrics.coco.recall import COCORecall


class CocoRecallTest(tf.test.TestCase):
    def test_no_boxes_found(self):
        category = -1
        # y_true = [images, bboxes, 5]
        y_true = tf.constant([[[100, 101, 200, 201, category]]], dtype=tf.float32)
        # y_pred = [images, bboxes, 6], the extra dim is confidence
        y_pred = tf.constant(
            [[[1000, 1100, 1200, 1101, category, 1.0]]], dtype=tf.float32
        )

        metric = COCORecall(categories=[-1])
        metric.update_state(y_true, y_pred)

        self.assertEqual(metric.result(), 0.0)

    def test_tf_function_no_boxes_found(self):
        category = -1
        # y_true = [images, bboxes, 5]
        y_true = tf.constant([[[100, 101, 200, 201, category]]], dtype=tf.float32)
        # y_pred = [images, bboxes, 6], the extra dim is confidence
        y_pred = tf.constant(
            [[[1000, 1100, 1200, 1101, category, 1.0]]], dtype=tf.float32
        )

        metric = COCORecall(categories=[-1])
        metric.update_state = tf.function(metric.update_state)
        metric.update_state(y_true, y_pred)

        self.assertEqual(metric.result(), 0.0)

    def test_every_box_found(self):
        category = -1
        # y_true = [images, bboxes, 5]
        y_true = tf.constant([[[100, 101, 200, 201, category]]], dtype=tf.float32)
        # y_pred = [images, bboxes, 6], the extra dim is confidence
        y_pred = tf.constant([[[100, 101, 200, 201, category, 1.0]]], dtype=tf.float32)

        metric = COCORecall(categories=[-1])
        metric.update_state(y_true, y_pred)

        self.assertEqual(metric.result(), 1.0)

    def test_recall_two_update_states(self):
        category = -1
        # y_true = [images, bboxes, 5]
        y_true = tf.constant([[[100, 101, 200, 201, category]]], dtype=tf.float32)
        # y_pred = [images, bboxes, 6], the extra dim is confidence
        y_pred = tf.constant([[[100, 101, 200, 201, category, 1.0]]], dtype=tf.float32)

        metric = COCORecall(categories=[-1])
        metric.update_state(y_true, y_pred)
        y_pred = tf.constant(
            [[[1000, 1100, 1200, 1101, category, 1.0]]], dtype=tf.float32
        )
        metric.update_state(y_true, y_pred)

        self.assertEqual(metric.result(), 0.5)

    def test_half_of_boxes_found(self):
        category = -1
        y_true = tf.constant(
            [[[100, 101, 200, 201, category], [200, 201, 300, 301, category]]],
            dtype=tf.float32,
        )
        y_pred = tf.constant([[[100, 101, 200, 201, category, 1.0]]], dtype=tf.float32)

        metric = COCORecall(categories=[-1])
        metric.update_state(y_true, y_pred)

        self.assertEqual(metric.result(), 0.5)

    def test_iou_thrs(self):
        category = -1
        y_true = tf.constant([[[100, 101, 200, 201, category]]], dtype=tf.float32)
        y_pred = tf.constant([[[130, 131, 230, 231, category, 1.0]]], dtype=tf.float32)

        metric = COCORecall(iou_thresholds=([0.05, 0.95]), categories=[-1])
        metric.update_state(y_true, y_pred)

        # the recall should average over iou_thresholds to get 0.5, one hit one miss
        self.assertEqual(metric.result(), 0.5)

    def test_missing_class(self):
        category = 1
        y_true = tf.constant([[[100, 101, 200, 201, category]]], dtype=tf.float32)
        y_pred = tf.constant([[[100, 101, 200, 201, category, 1.0]]], dtype=tf.float32)

        metric = COCORecall(categories=[1, 2])
        metric.update_state(y_true, y_pred)

        self.assertEqual(metric.result(), 1.0)

    def test_two_categories_one_hit_one_duplicate(self):
        y_true = tf.constant(
            [[[100, 101, 200, 201, 1], [100, 101, 200, 201, 2],]], dtype=tf.float32
        )
        # here we have a duplicate detection, and we miss the one of class 2
        y_pred = tf.constant(
            [[[100, 101, 200, 201, 1, 1.0], [100, 101, 202, 201, 1, 0.08]]],
            dtype=tf.float32,
        )

        metric = COCORecall(categories=[1, 2])
        metric.update_state(y_true, y_pred)

        self.assertEqual(metric.result(), 0.5)

    def test_two_categories(self):
        y_true = tf.constant(
            [[[100, 101, 200, 201, 1], [100, 101, 200, 201, 2],]], dtype=tf.float32
        )
        y_pred = tf.constant(
            [[[100, 101, 200, 201, 1, 1.0], [100, 101, 200, 201, 2, 1.0]]],
            dtype=tf.float32,
        )

        # let's also include a missing class
        metric = COCORecall(categories=[1, 2, 3])
        metric.update_state(y_true, y_pred)

        self.assertEqual(metric.result(), 1.0)

    def test_two_classes_two_images(self):
        y_true = tf.constant(
            [[[100, 101, 200, 201, 1]], [[100, 101, 200, 201, 2]]], dtype=tf.float32
        )
        y_pred = tf.constant(
            [[[100, 101, 200, 201, 1, 1.0]], [[100, 101, 200, 201, 2, 1.0]]],
            dtype=tf.float32,
        )

        metric = COCORecall(categories=[1, 2])
        metric.update_state(y_true, y_pred)

        self.assertEqual(metric.result(), 1.0)

    def test_two_classes_two_images_wrong_category(self):
        y_true = tf.constant(
            [[[100, 101, 200, 201, 1]], [[100, 101, 200, 201, 2]]], dtype=tf.float32
        )
        y_pred = tf.constant(
            [[[100, 101, 200, 201, 2, 1.0]], [[100, 101, 200, 201, 1, 1.0]]],
            dtype=tf.float32,
        )

        metric = COCORecall(categories=[1, 2])
        metric.update_state(y_true, y_pred)

        self.assertEqual(metric.result(), 0.0)

    def test_wrong_class(self):
        # y_true = [images, bboxes, 5]
        y_true = tf.constant([[[100, 101, 200, 201, 1]]], dtype=tf.float32)
        # y_pred = [images, bboxes, 6], the extra dim is confidence
        y_pred = tf.constant([[[100, 101, 200, 201, 2, 1.0]]], dtype=tf.float32)

        metric = COCORecall(categories=[1, 2])
        metric.update_state(y_true, y_pred)

        self.assertEqual(metric.result(), 0.0)


if __name__ == "__main__":
    googletest.main()
