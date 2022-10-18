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

import tensorflow as tf

from keras_cv.layers.object_detection.roi_sampler import _ROISampler
from keras_cv.ops.box_matcher import ArgmaxBoxMatcher


class ROISamplerTest(tf.test.TestCase):
    def test_roi_sampler(self):
        box_matcher = ArgmaxBoxMatcher(thresholds=[0.3], match_values=[-1, 1])
        roi_sampler = _ROISampler(
            bounding_box_format="xyxy",
            roi_matcher=box_matcher,
            positive_fraction=0.5,
            num_sampled_rois=2,
            append_gt_boxes=False,
        )
        rois = tf.constant(
            [[0, 0, 5, 5], [2.5, 2.5, 7.5, 7.5], [5, 5, 10, 10], [7.5, 7.5, 12.5, 12.5]]
        )
        rois = rois[tf.newaxis, ...]
        # the 3rd box will generate 0 IOUs and not sampled.
        gt_boxes = tf.constant(
            [[10, 10, 15, 15], [2.5, 2.5, 7.5, 7.5], [-1, -1, -1, -1]]
        )
        gt_boxes = gt_boxes[tf.newaxis, ...]
        gt_classes = tf.constant([[2, 10, -1]], dtype=tf.int32)
        gt_classes = gt_classes[..., tf.newaxis]
        _, sampled_gt_boxes, _, sampled_gt_classes, _ = roi_sampler(
            rois, gt_boxes, gt_classes
        )
        # given we only choose 1 positive sample, and `append_labesl` is False,
        # only the 2nd ROI is chosen.
        expected_gt_boxes = tf.constant([[0.0, 0.0, 0, 0.0], [0.0, 0.0, 0, 0.0]])
        expected_gt_boxes = expected_gt_boxes[tf.newaxis, ...]
        # only the 2nd ROI is chosen, and the negative ROI is mapped to 0.
        expected_gt_classes = tf.constant([[10], [0]], dtype=tf.int32)
        expected_gt_classes = expected_gt_classes[tf.newaxis, ...]
        self.assertAllClose(
            tf.reduce_max(expected_gt_boxes), tf.reduce_max(sampled_gt_boxes)
        )
        self.assertAllClose(
            tf.reduce_min(expected_gt_classes), tf.reduce_min(sampled_gt_classes)
        )

    def test_roi_sampler_small_threshold(self):
        box_matcher = ArgmaxBoxMatcher(thresholds=[0.1], match_values=[-1, 1])
        roi_sampler = _ROISampler(
            bounding_box_format="xyxy",
            roi_matcher=box_matcher,
            positive_fraction=0.5,
            num_sampled_rois=2,
            append_gt_boxes=False,
        )
        rois = tf.constant(
            [[0, 0, 5, 5], [2.5, 2.5, 7.5, 7.5], [5, 5, 10, 10], [7.5, 7.5, 12.5, 12.5]]
        )
        rois = rois[tf.newaxis, ...]
        # the 3rd box will generate 0 IOUs and not sampled.
        gt_boxes = tf.constant(
            [[10, 10, 15, 15], [2.6, 2.6, 7.6, 7.6], [-1, -1, -1, -1]]
        )
        gt_boxes = gt_boxes[tf.newaxis, ...]
        gt_classes = tf.constant([[2, 10, -1]], dtype=tf.int32)
        gt_classes = gt_classes[..., tf.newaxis]
        sampled_rois, sampled_gt_boxes, _, sampled_gt_classes, _ = roi_sampler(
            rois, gt_boxes, gt_classes
        )
        # given we only choose 1 positive sample, and `append_labesl` is False,
        # only the 2nd ROI is chosen. No negative samples exist given we
        # select positive_threshold to be 0.1. (the minimum IOU is 1/7)
        # given num_sampled_rois=2, it selects the 1st ROI as well.
        expected_rois = tf.constant([[5, 5, 10, 10], [0.0, 0.0, 5.0, 5.0]])
        expected_rois = expected_rois[tf.newaxis, ...]
        # all ROIs are matched to the 2nd gt box.
        # the boxes are encoded by dimensions, so the result is
        # tx, ty = (5.1 - 5.0) / 5 = 0.02, tx, ty = (5.1 - 2.5) / 5 = 0.52
        # then divide by 0.1 as box variance.
        expected_gt_boxes = (
            tf.constant([[0.02, 0.02, 0.0, 0.0], [0.52, 0.52, 0.0, 0.0]]) / 0.1
        )
        expected_gt_boxes = expected_gt_boxes[tf.newaxis, ...]
        # only the 2nd ROI is chosen, and the negative ROI is mapped to 0.
        expected_gt_classes = tf.constant([[10], [10]], dtype=tf.int32)
        expected_gt_classes = expected_gt_classes[tf.newaxis, ...]
        self.assertAllClose(
            tf.reduce_max(expected_rois, 1), tf.reduce_max(sampled_rois, 1)
        )
        self.assertAllClose(
            tf.reduce_max(expected_gt_boxes, 1), tf.reduce_max(sampled_gt_boxes, 1)
        )
        self.assertAllClose(expected_gt_classes, sampled_gt_classes)

    def test_roi_sampler_large_threshold(self):
        # the 2nd roi and 2nd gt box has IOU of 0.923, setting positive_threshold to 0.95 to ignore it
        box_matcher = ArgmaxBoxMatcher(thresholds=[0.95], match_values=[-1, 1])
        roi_sampler = _ROISampler(
            bounding_box_format="xyxy",
            roi_matcher=box_matcher,
            positive_fraction=0.5,
            num_sampled_rois=2,
            append_gt_boxes=False,
        )
        rois = tf.constant(
            [[0, 0, 5, 5], [2.5, 2.5, 7.5, 7.5], [5, 5, 10, 10], [7.5, 7.5, 12.5, 12.5]]
        )
        rois = rois[tf.newaxis, ...]
        # the 3rd box will generate 0 IOUs and not sampled.
        gt_boxes = tf.constant(
            [[10, 10, 15, 15], [2.6, 2.6, 7.6, 7.6], [-1, -1, -1, -1]]
        )
        gt_boxes = gt_boxes[tf.newaxis, ...]
        gt_classes = tf.constant([[2, 10, -1]], dtype=tf.int32)
        gt_classes = gt_classes[..., tf.newaxis]
        _, sampled_gt_boxes, _, sampled_gt_classes, _ = roi_sampler(
            rois, gt_boxes, gt_classes
        )
        # all ROIs are negative matches, so they are mapped to 0.
        expected_gt_boxes = tf.zeros([1, 2, 4], dtype=tf.float32)
        # only the 2nd ROI is chosen, and the negative ROI is mapped to 0.
        expected_gt_classes = tf.constant([[0], [0]], dtype=tf.int32)
        expected_gt_classes = expected_gt_classes[tf.newaxis, ...]
        # self.assertAllClose(expected_rois, sampled_rois)
        self.assertAllClose(expected_gt_boxes, sampled_gt_boxes)
        self.assertAllClose(expected_gt_classes, sampled_gt_classes)

    def test_roi_sampler_large_threshold_custom_bg_class(self):
        # the 2nd roi and 2nd gt box has IOU of 0.923, setting positive_threshold to 0.95 to ignore it
        box_matcher = ArgmaxBoxMatcher(thresholds=[0.95], match_values=[-1, 1])
        roi_sampler = _ROISampler(
            bounding_box_format="xyxy",
            roi_matcher=box_matcher,
            positive_fraction=0.5,
            background_class=-1,
            num_sampled_rois=2,
            append_gt_boxes=False,
        )
        rois = tf.constant(
            [[0, 0, 5, 5], [2.5, 2.5, 7.5, 7.5], [5, 5, 10, 10], [7.5, 7.5, 12.5, 12.5]]
        )
        rois = rois[tf.newaxis, ...]
        # the 3rd box will generate 0 IOUs and not sampled.
        gt_boxes = tf.constant(
            [[10, 10, 15, 15], [2.6, 2.6, 7.6, 7.6], [-1, -1, -1, -1]]
        )
        gt_boxes = gt_boxes[tf.newaxis, ...]
        gt_classes = tf.constant([[2, 10, -1]], dtype=tf.int32)
        gt_classes = gt_classes[..., tf.newaxis]
        _, sampled_gt_boxes, _, sampled_gt_classes, _ = roi_sampler(
            rois, gt_boxes, gt_classes
        )
        # all ROIs are negative matches, so they are mapped to 0.
        expected_gt_boxes = tf.zeros([1, 2, 4], dtype=tf.float32)
        # only the 2nd ROI is chosen, and the negative ROI is mapped to -1 from customization.
        expected_gt_classes = tf.constant([[-1], [-1]], dtype=tf.int32)
        expected_gt_classes = expected_gt_classes[tf.newaxis, ...]
        # self.assertAllClose(expected_rois, sampled_rois)
        self.assertAllClose(expected_gt_boxes, sampled_gt_boxes)
        self.assertAllClose(expected_gt_classes, sampled_gt_classes)

    def test_roi_sampler_large_threshold_append_gt_boxes(self):
        # the 2nd roi and 2nd gt box has IOU of 0.923, setting positive_threshold to 0.95 to ignore it
        box_matcher = ArgmaxBoxMatcher(thresholds=[0.95], match_values=[-1, 1])
        roi_sampler = _ROISampler(
            bounding_box_format="xyxy",
            roi_matcher=box_matcher,
            positive_fraction=0.5,
            num_sampled_rois=2,
            append_gt_boxes=True,
        )
        rois = tf.constant(
            [[0, 0, 5, 5], [2.5, 2.5, 7.5, 7.5], [5, 5, 10, 10], [7.5, 7.5, 12.5, 12.5]]
        )
        rois = rois[tf.newaxis, ...]
        # the 3rd box will generate 0 IOUs and not sampled.
        gt_boxes = tf.constant(
            [[10, 10, 15, 15], [2.6, 2.6, 7.6, 7.6], [-1, -1, -1, -1]]
        )
        gt_boxes = gt_boxes[tf.newaxis, ...]
        gt_classes = tf.constant([[2, 10, -1]], dtype=tf.int32)
        gt_classes = gt_classes[..., tf.newaxis]
        _, sampled_gt_boxes, _, sampled_gt_classes, _ = roi_sampler(
            rois, gt_boxes, gt_classes
        )
        # the selected gt boxes should be [0, 0, 0, 0], and [10, 10, 15, 15]
        # but the 2nd will be encoded to 0.
        self.assertAllClose(tf.reduce_min(sampled_gt_boxes), 0)
        self.assertAllClose(tf.reduce_max(sampled_gt_boxes), 0)
        # the selected gt classes should be [0, 2 or 10]
        self.assertAllLessEqual(tf.reduce_max(sampled_gt_classes), 10)
        self.assertAllGreaterEqual(tf.reduce_min(sampled_gt_classes), 0)

    def test_roi_sampler_large_num_sampled_rois(self):
        box_matcher = ArgmaxBoxMatcher(thresholds=[0.95], match_values=[-1, 1])
        roi_sampler = _ROISampler(
            bounding_box_format="xyxy",
            roi_matcher=box_matcher,
            positive_fraction=0.5,
            num_sampled_rois=200,
            append_gt_boxes=True,
        )
        rois = tf.constant(
            [[0, 0, 5, 5], [2.5, 2.5, 7.5, 7.5], [5, 5, 10, 10], [7.5, 7.5, 12.5, 12.5]]
        )
        rois = rois[tf.newaxis, ...]
        # the 3rd box will generate 0 IOUs and not sampled.
        gt_boxes = tf.constant(
            [[10, 10, 15, 15], [2.6, 2.6, 7.6, 7.6], [-1, -1, -1, -1]]
        )
        gt_boxes = gt_boxes[tf.newaxis, ...]
        gt_classes = tf.constant([[2, 10, -1]], dtype=tf.int32)
        gt_classes = gt_classes[..., tf.newaxis]
        with self.assertRaisesRegex(ValueError, "must be less than"):
            _, _, _ = roi_sampler(rois, gt_boxes, gt_classes)

    def test_serialization(self):
        box_matcher = ArgmaxBoxMatcher(thresholds=[0.95], match_values=[-1, 1])
        roi_sampler = _ROISampler(
            bounding_box_format="xyxy",
            roi_matcher=box_matcher,
            positive_fraction=0.5,
            num_sampled_rois=200,
            append_gt_boxes=True,
        )
        sampler_config = roi_sampler.get_config()
        new_sampler = _ROISampler.from_config(sampler_config)
        self.assertAllEqual(new_sampler.roi_matcher.match_values, [-1, 1])
