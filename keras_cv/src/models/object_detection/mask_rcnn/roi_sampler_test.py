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


import numpy as np
import pytest
from absl.testing import parameterized

from keras_cv.src.backend import ops
from keras_cv.src.backend.config import keras_3
from keras_cv.src.layers.object_detection.box_matcher import BoxMatcher
from keras_cv.src.models.object_detection.mask_rcnn.roi_sampler import (
    ROISampler,
)
from keras_cv.src.tests.test_case import TestCase


class ROISamplerTest(TestCase):
    @parameterized.parameters((0,), (1,), (2,))
    @pytest.mark.skipif(not keras_3(), reason="disabling test for Keras 2")
    def test_roi_sampler(self, mask_value):
        box_matcher = BoxMatcher(thresholds=[0.3], match_values=[-1, 1])
        roi_sampler = ROISampler(
            roi_bounding_box_format="xyxy",
            gt_bounding_box_format="xyxy",
            roi_matcher=box_matcher,
            positive_fraction=0.5,
            num_sampled_rois=2,
            append_gt_boxes=False,
        )
        rois = np.array(
            [
                [0, 0, 5, 5],
                [2.5, 2.5, 7.5, 7.5],
                [5, 5, 10, 10],
                [7.5, 7.5, 12.5, 12.5],
            ]
        )
        rois = rois[np.newaxis, ...]
        # the 3rd box will generate 0 IOUs and not sampled.
        gt_boxes = np.array(
            [[10, 10, 15, 15], [2.5, 2.5, 7.5, 7.5], [-1, -1, -1, -1]]
        )
        gt_boxes = gt_boxes[np.newaxis, ...]
        gt_classes = np.array([[2, 10, -1]], dtype=np.int32)
        gt_classes = gt_classes[..., np.newaxis]
        gt_masks = mask_value * np.ones((1, 20, 20), dtype=np.uint8)
        _, sampled_gt_boxes, _, sampled_gt_classes, _, sampled_gt_masks, _ = (
            roi_sampler(rois, gt_boxes, gt_classes, gt_masks)
        )
        # given we only choose 1 positive sample, and `append_label` is False,
        # only the 2nd ROI is chosen.
        expected_gt_boxes = np.array([[0.0, 0.0, 0, 0.0], [0.0, 0.0, 0, 0.0]])
        expected_gt_boxes = expected_gt_boxes[np.newaxis, ...]
        # only the 2nd ROI is chosen, and the negative ROI is mapped to 0.
        expected_gt_classes = np.array([[10], [0]], dtype=np.int32)
        expected_gt_classes = expected_gt_classes[np.newaxis, ...]
        self.assertAllClose(
            np.max(expected_gt_boxes),
            np.max(ops.convert_to_numpy(sampled_gt_boxes)),
        )
        self.assertAllClose(
            np.min(expected_gt_classes),
            np.min(ops.convert_to_numpy(sampled_gt_classes)),
        )
        # the sampled mask is only set to 1 if the ground truth
        # mask indicates object 2
        sampled_index = ops.where(sampled_gt_classes[0, :, 0] == 10)[0][0]
        self.assertAllClose(
            sampled_gt_masks[0, sampled_index],
            (mask_value == 2) * np.ones((14, 14)),
        )

    @pytest.mark.skipif(not keras_3(), reason="disabling test for Keras 2")
    def test_roi_sampler_small_threshold(self):
        self.skipTest(
            "TODO: resolving flaky test, https://github.com/keras-team/keras-cv/issues/2336"  # noqa
        )
        box_matcher = BoxMatcher(thresholds=[0.1], match_values=[-1, 1])
        roi_sampler = ROISampler(
            roi_bounding_box_format="xyxy",
            gt_bounding_box_format="xyxy",
            roi_matcher=box_matcher,
            positive_fraction=0.5,
            num_sampled_rois=2,
            append_gt_boxes=False,
        )
        rois = np.array(
            [
                [0, 0, 5, 5],
                [2.5, 2.5, 7.5, 7.5],
                [5, 5, 10, 10],
                [7.5, 7.5, 12.5, 12.5],
            ]
        )
        rois = rois[np.newaxis, ...]
        # the 3rd box will generate 0 IOUs and not sampled.
        gt_boxes = np.array(
            [[10, 10, 15, 15], [2.6, 2.6, 7.6, 7.6], [-1, -1, -1, -1]]
        )
        gt_boxes = gt_boxes[np.newaxis, ...]
        gt_classes = np.array([[2, 10, -1]], dtype=np.int32)
        gt_classes = gt_classes[..., np.newaxis]
        sampled_rois, sampled_gt_boxes, _, sampled_gt_classes, _ = roi_sampler(
            rois, gt_boxes, gt_classes
        )
        # given we only choose 1 positive sample, and `append_label` is False,
        # only the 2nd ROI is chosen. No negative samples exist given we
        # select positive_threshold to be 0.1. (the minimum IOU is 1/7)
        # given num_sampled_rois=2, it selects the 1st ROI as well.
        expected_rois = np.array([[5, 5, 10, 10], [0.0, 0.0, 5.0, 5.0]])
        expected_rois = expected_rois[np.newaxis, ...]
        # all ROIs are matched to the 2nd gt box.
        # the boxes are encoded by dimensions, so the result is
        # tx, ty = (5.1 - 5.0) / 5 = 0.02, tx, ty = (5.1 - 2.5) / 5 = 0.52
        # then divide by 0.1 as box variance.
        expected_gt_boxes = (
            np.array([[0.02, 0.02, 0.0, 0.0], [0.52, 0.52, 0.0, 0.0]]) / 0.1
        )
        expected_gt_boxes = expected_gt_boxes[np.newaxis, ...]
        # only the 2nd ROI is chosen, and the negative ROI is mapped to 0.
        expected_gt_classes = np.array([[10], [10]], dtype=np.int32)
        expected_gt_classes = expected_gt_classes[np.newaxis, ...]
        self.assertAllClose(np.max(expected_rois, 1), np.max(sampled_rois, 1))
        self.assertAllClose(
            np.max(expected_gt_boxes, 1),
            np.max(sampled_gt_boxes, 1),
        )
        self.assertAllClose(expected_gt_classes, sampled_gt_classes)

    @pytest.mark.skipif(not keras_3(), reason="disabling test for Keras 2")
    def test_roi_sampler_large_threshold(self):
        # the 2nd roi and 2nd gt box has IOU of 0.923, setting
        # positive_threshold to 0.95 to ignore it.
        box_matcher = BoxMatcher(thresholds=[0.95], match_values=[-1, 1])
        roi_sampler = ROISampler(
            roi_bounding_box_format="xyxy",
            gt_bounding_box_format="xyxy",
            roi_matcher=box_matcher,
            positive_fraction=0.5,
            num_sampled_rois=2,
            append_gt_boxes=False,
        )
        rois = np.array(
            [
                [0, 0, 5, 5],
                [2.5, 2.5, 7.5, 7.5],
                [5, 5, 10, 10],
                [7.5, 7.5, 12.5, 12.5],
            ]
        )
        rois = rois[np.newaxis, ...]
        # the 3rd box will generate 0 IOUs and not sampled.
        gt_boxes = np.array(
            [[10, 10, 15, 15], [2.6, 2.6, 7.6, 7.6], [-1, -1, -1, -1]]
        )
        gt_boxes = gt_boxes[np.newaxis, ...]
        gt_classes = np.array([[2, 10, -1]], dtype=np.int32)
        gt_classes = gt_classes[..., np.newaxis]
        _, sampled_gt_boxes, _, sampled_gt_classes, _ = roi_sampler(
            rois, gt_boxes, gt_classes
        )
        # all ROIs are negative matches, so they are mapped to 0.
        expected_gt_boxes = np.zeros([1, 2, 4], dtype=np.float32)
        # only the 2nd ROI is chosen, and the negative ROI is mapped to 0.
        expected_gt_classes = np.array([[0], [0]], dtype=np.int32)
        expected_gt_classes = expected_gt_classes[np.newaxis, ...]
        # self.assertAllClose(expected_rois, sampled_rois)
        self.assertAllClose(expected_gt_boxes, sampled_gt_boxes)
        self.assertAllClose(expected_gt_classes, sampled_gt_classes)

    @pytest.mark.skipif(not keras_3(), reason="disabling test for Keras 2")
    def test_roi_sampler_large_threshold_custom_bg_class(self):
        # the 2nd roi and 2nd gt box has IOU of 0.923, setting
        # positive_threshold to 0.95 to ignore it.
        box_matcher = BoxMatcher(thresholds=[0.95], match_values=[-1, 1])
        roi_sampler = ROISampler(
            roi_bounding_box_format="xyxy",
            gt_bounding_box_format="xyxy",
            roi_matcher=box_matcher,
            positive_fraction=0.5,
            background_class=-1,
            num_sampled_rois=2,
            append_gt_boxes=False,
        )
        rois = np.array(
            [
                [0, 0, 5, 5],
                [2.5, 2.5, 7.5, 7.5],
                [5, 5, 10, 10],
                [7.5, 7.5, 12.5, 12.5],
            ]
        )
        rois = rois[np.newaxis, ...]
        # the 3rd box will generate 0 IOUs and not sampled.
        gt_boxes = np.array(
            [[10, 10, 15, 15], [2.6, 2.6, 7.6, 7.6], [-1, -1, -1, -1]]
        )
        gt_boxes = gt_boxes[np.newaxis, ...]
        gt_classes = np.array([[2, 10, -1]], dtype=np.int32)
        gt_classes = gt_classes[..., np.newaxis]
        _, sampled_gt_boxes, _, sampled_gt_classes, _ = roi_sampler(
            rois, gt_boxes, gt_classes
        )
        # all ROIs are negative matches, so they are mapped to 0.
        expected_gt_boxes = np.zeros([1, 2, 4], dtype=np.float32)
        # only the 2nd ROI is chosen, and the negative ROI is mapped to -1 from
        # customization.
        expected_gt_classes = np.array([[-1], [-1]], dtype=np.int32)
        expected_gt_classes = expected_gt_classes[np.newaxis, ...]
        # self.assertAllClose(expected_rois, sampled_rois)
        self.assertAllClose(expected_gt_boxes, sampled_gt_boxes)
        self.assertAllClose(expected_gt_classes, sampled_gt_classes)

    @pytest.mark.skipif(not keras_3(), reason="disabling test for Keras 2")
    def test_roi_sampler_large_threshold_append_gt_boxes(self):
        # the 2nd roi and 2nd gt box has IOU of 0.923, setting
        # positive_threshold to 0.95 to ignore it.
        box_matcher = BoxMatcher(thresholds=[0.95], match_values=[-1, 1])
        roi_sampler = ROISampler(
            roi_bounding_box_format="xyxy",
            gt_bounding_box_format="xyxy",
            roi_matcher=box_matcher,
            positive_fraction=0.5,
            num_sampled_rois=2,
            append_gt_boxes=True,
        )
        rois = np.array(
            [
                [0, 0, 5, 5],
                [2.5, 2.5, 7.5, 7.5],
                [5, 5, 10, 10],
                [7.5, 7.5, 12.5, 12.5],
            ]
        )
        rois = rois[np.newaxis, ...]
        # the 3rd box will generate 0 IOUs and not sampled.
        gt_boxes = np.array(
            [[10, 10, 15, 15], [2.6, 2.6, 7.6, 7.6], [-1, -1, -1, -1]]
        )
        gt_boxes = gt_boxes[np.newaxis, ...]
        gt_classes = np.array([[2, 10, -1]], dtype=np.int32)
        gt_classes = gt_classes[..., np.newaxis]
        _, sampled_gt_boxes, _, sampled_gt_classes, _ = roi_sampler(
            rois, gt_boxes, gt_classes
        )
        # the selected gt boxes should be [0, 0, 0, 0], and [10, 10, 15, 15]
        # but the 2nd will be encoded to 0.
        self.assertAllClose(np.min(ops.convert_to_numpy(sampled_gt_boxes)), 0)
        self.assertAllClose(np.max(ops.convert_to_numpy(sampled_gt_boxes)), 0)
        # the selected gt classes should be [0, 2 or 10]
        self.assertAllLessEqual(
            np.max(ops.convert_to_numpy(sampled_gt_classes)), 10
        )
        self.assertAllGreaterEqual(
            np.min(ops.convert_to_numpy(sampled_gt_classes)), 0
        )

    @pytest.mark.skipif(not keras_3(), reason="disabling test for Keras 2")
    def test_roi_sampler_large_num_sampled_rois(self):
        box_matcher = BoxMatcher(thresholds=[0.95], match_values=[-1, 1])
        roi_sampler = ROISampler(
            roi_bounding_box_format="xyxy",
            gt_bounding_box_format="xyxy",
            roi_matcher=box_matcher,
            positive_fraction=0.5,
            num_sampled_rois=200,
            append_gt_boxes=True,
        )
        rois = np.array(
            [
                [0, 0, 5, 5],
                [2.5, 2.5, 7.5, 7.5],
                [5, 5, 10, 10],
                [7.5, 7.5, 12.5, 12.5],
            ]
        )
        rois = rois[np.newaxis, ...]
        # the 3rd box will generate 0 IOUs and not sampled.
        gt_boxes = np.array(
            [[10, 10, 15, 15], [2.6, 2.6, 7.6, 7.6], [-1, -1, -1, -1]]
        )
        gt_boxes = gt_boxes[np.newaxis, ...]
        gt_classes = np.array([[2, 10, -1]], dtype=np.int32)
        gt_classes = gt_classes[..., np.newaxis]
        with self.assertRaisesRegex(ValueError, "must be less than"):
            _, _, _ = roi_sampler(rois, gt_boxes, gt_classes)

    @pytest.mark.skipif(not keras_3(), reason="disabling test for Keras 2")
    def test_serialization(self):
        box_matcher = BoxMatcher(thresholds=[0.95], match_values=[-1, 1])
        roi_sampler = ROISampler(
            roi_bounding_box_format="xyxy",
            gt_bounding_box_format="xyxy",
            roi_matcher=box_matcher,
            positive_fraction=0.5,
            num_sampled_rois=200,
            append_gt_boxes=True,
        )
        sampler_config = roi_sampler.get_config()
        new_sampler = ROISampler.from_config(sampler_config)
        self.assertAllEqual(new_sampler.roi_matcher.match_values, [-1, 1])
