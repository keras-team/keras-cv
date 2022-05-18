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
"""Tests for util functions."""

import tensorflow as tf

from keras_cv import bounding_box
from keras_cv.metrics.coco import utils
from keras_cv.utils import iou as iou_lib


class UtilTest(tf.test.TestCase):
    def test_filter_bounding_boxes_empty(self):
        # set of bounding_boxes
        y_pred = tf.stack([_dummy_bounding_box(category=1)])
        result = utils.filter_boxes(y_pred, 2, axis=bounding_box.XYXY.CLASS)

        self.assertEqual(result.shape[0], 0)

    def test_bounding_box_area(self):
        boxes = tf.constant([[0, 0, 100, 100]], dtype=tf.float32)
        areas = utils.bounding_box_area(boxes)
        self.assertAllClose(areas, tf.constant((10000.0,)))

    def test_filter_bounding_boxes(self):
        # set of bounding_boxes
        y_pred = tf.stack(
            [_dummy_bounding_box(category=1), _dummy_bounding_box(category=2)]
        )
        result = utils.filter_boxes(y_pred, 2, axis=bounding_box.XYXY.CLASS)

        self.assertAllClose(result, tf.stack([_dummy_bounding_box(category=2)]))

    def test_to_sentinel_padded_bounding_box_tensor(self):
        box_set1 = tf.stack([_dummy_bounding_box(), _dummy_bounding_box()])
        box_set2 = tf.stack([_dummy_bounding_box()])
        boxes = [box_set1, box_set2]
        bounding_box_tensor = utils.to_sentinel_padded_bounding_box_tensor(boxes)
        self.assertAllClose(
            bounding_box_tensor[1, 1],
            -tf.ones(
                6,
            ),
        )

    def test_filter_out_sentinels(self):
        # set of bounding_boxes
        y_pred = tf.stack(
            [_dummy_bounding_box(category=1), _dummy_bounding_box(category=-1)]
        )
        result = utils.filter_out_sentinels(y_pred)

        self.assertAllClose(result, tf.stack([_dummy_bounding_box(category=1)]))

    def test_end_to_end_sentinel_filtering(self):
        box_set1 = tf.stack([_dummy_bounding_box(), _dummy_bounding_box()])
        box_set2 = tf.stack([_dummy_bounding_box()])
        boxes = [box_set1, box_set2]
        bounding_box_tensor = utils.to_sentinel_padded_bounding_box_tensor(boxes)

        self.assertAllClose(
            utils.filter_out_sentinels(bounding_box_tensor[0]), box_set1
        )
        self.assertAllClose(
            utils.filter_out_sentinels(bounding_box_tensor[1]), box_set2
        )

    def test_match_boxes(self):
        y_pred = tf.stack(
            [
                _dummy_bounding_box(0.1),
                _dummy_bounding_box(0.9),
                _dummy_bounding_box(0.4),
            ]
        )
        y_true = tf.stack(
            [
                _dummy_bounding_box(0.1),
                _dummy_bounding_box(0.9),
                _dummy_bounding_box(0.4),
                _dummy_bounding_box(0.2),
            ]
        )

        ious = iou_lib.compute_ious_for_image(y_true, y_pred)
        self.assertEqual(utils.match_boxes(ious, 0.5).shape, [3])

    def test_sort_bounding_boxes_unsorted_list(self):
        y_pred = tf.expand_dims(
            tf.stack(
                [
                    _dummy_bounding_box(0.1),
                    _dummy_bounding_box(0.9),
                    _dummy_bounding_box(0.4),
                    _dummy_bounding_box(0.2),
                ]
            ),
            axis=0,
        )
        want = tf.expand_dims(
            tf.stack(
                [
                    _dummy_bounding_box(0.9),
                    _dummy_bounding_box(0.4),
                    _dummy_bounding_box(0.2),
                    _dummy_bounding_box(0.1),
                ]
            ),
            axis=0,
        )
        y_sorted = utils.sort_bounding_boxes(y_pred, bounding_box.XYXY.CONFIDENCE)
        self.assertAllClose(y_sorted, want)

    def test_sort_bounding_boxes_empty_list(self):
        y_pred = tf.stack([])
        y_sorted = utils.sort_bounding_boxes(y_pred)
        self.assertAllClose(y_pred, y_sorted)


def _dummy_bounding_box(confidence=0.0, category=0):
    """returns a bounding_box dummy with all 0 values, except for confidence."""
    return tf.constant([0, 0, 0, 0, category, confidence])
