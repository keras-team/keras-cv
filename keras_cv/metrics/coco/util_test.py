"""Tests for util functions."""

import numpy as np
import tensorflow as tf

from keras_cv.metrics.coco import util
from keras_cv.util import bbox


class UtilTest(tf.test.TestCase):
    def test_filter_bboxes_empty(self):
        # set of bboxes
        y_pred = tf.stack([_dummy_bbox(category=1)])
        result = util.filter_boxes(y_pred, 2, axis=bbox.CLASS)

        self.assertEqual(result.shape[0], 0)

    def test_bbox_area(self):
        boxes = tf.constant([[0, 0, 100, 100]], dtype=tf.float32)
        areas = util.bbox_area(boxes)
        self.assertAllClose(areas, tf.constant((10000.0,)))

    def test_filter_bboxes(self):
        # set of bboxes
        y_pred = tf.stack([_dummy_bbox(category=1), _dummy_bbox(category=2)])
        result = util.filter_boxes(y_pred, 2, axis=bbox.CLASS)

        self.assertAllClose(result, tf.stack([_dummy_bbox(category=2)]))

    def test_to_sentinel_padded_bbox_tensor(self):
        box_set1 = tf.stack([_dummy_bbox(), _dummy_bbox()])
        box_set2 = tf.stack([_dummy_bbox()])
        boxes = [box_set1, box_set2]
        bbox_tensor = util.to_sentinel_padded_bbox_tensor(boxes)
        self.assertAllClose(
            bbox_tensor[1, 1], -tf.ones(6,),
        )

    def test_filter_out_sentinels(self):
        # set of bboxes
        y_pred = tf.stack([_dummy_bbox(category=1), _dummy_bbox(category=-1)])
        result = util.filter_out_sentinels(y_pred)

        self.assertAllClose(result, tf.stack([_dummy_bbox(category=1)]))

    def test_end_to_end_sentinel_filtering(self):
        box_set1 = tf.stack([_dummy_bbox(), _dummy_bbox()])
        box_set2 = tf.stack([_dummy_bbox()])
        boxes = [box_set1, box_set2]
        bbox_tensor = util.to_sentinel_padded_bbox_tensor(boxes)

        self.assertAllClose(util.filter_out_sentinels(bbox_tensor[0]), box_set1)
        self.assertAllClose(util.filter_out_sentinels(bbox_tensor[1]), box_set2)

    def test_sort_bboxes_unsorted_list(self):
        y_pred = tf.expand_dims(
            tf.stack(
                [_dummy_bbox(0.1), _dummy_bbox(0.9), _dummy_bbox(0.4), _dummy_bbox(0.2)]
            ),
            axis=0,
        )
        want = tf.expand_dims(
            tf.stack(
                [_dummy_bbox(0.9), _dummy_bbox(0.4), _dummy_bbox(0.2), _dummy_bbox(0.1)]
            ),
            axis=0,
        )
        y_sorted = util.sort_bboxes(y_pred, bbox.CONFIDENCE)
        self.assertAllClose(y_sorted, want)

    def test_sort_bboxes_empty_list(self):
        y_pred = tf.stack([])
        y_sorted = util.sort_bboxes(y_pred)
        self.assertAllClose(y_pred, y_sorted)


def _dummy_bbox(confidence=0.0, category=0):
    """returns a bbox dummy with all 0 values, except for confidence."""
    return tf.constant([0, 0, 0, 0, category, confidence])
