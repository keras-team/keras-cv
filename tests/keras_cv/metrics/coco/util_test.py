"""Tests for util functions."""

import numpy as np
import tensorflow as tf

from keras_cv.metrics.coco import util
from keras_cv.metrics.coco import bbox


class UtilTest(tf.test.TestCase):
    def test_sort_bboxes_unsorted_list(self):
        y_pred = tf.expand_dims(tf.stack(
            [_dummy_bbox(0.1), _dummy_bbox(0.9), _dummy_bbox(0.4), _dummy_bbox(0.2)]
        ), axis=0)
        want = tf.expand_dims(tf.stack(
            [_dummy_bbox(0.9), _dummy_bbox(0.4), _dummy_bbox(0.2), _dummy_bbox(0.1)]
        ), axis=0)
        y_sorted = util.sort_bboxes(y_pred, bbox.CONFIDENCE)
        self.assertAllClose(y_sorted, want)

    def test_sort_bboxes_empty_list(self):
        y_pred = tf.stack([])
        y_sorted = util.sort_bboxes(y_pred)
        self.assertAllClose(y_pred, y_sorted)


def _dummy_bbox(confidence):
    """returns a bbox dummy with all 0 values, except for confidence."""
    return tf.constant([0, 0, 0, 0, 0, confidence])
