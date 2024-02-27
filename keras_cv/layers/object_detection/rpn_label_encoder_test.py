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

from keras_cv.backend import ops
from keras_cv.layers.object_detection.rpn_label_encoder import _RpnLabelEncoder
from keras_cv.tests.test_case import TestCase


class RpnLabelEncoderTest(TestCase):
    def test_rpn_label_encoder(self):
        rpn_encoder = _RpnLabelEncoder(
            anchor_format="xyxy",
            ground_truth_box_format="xyxy",
            positive_threshold=0.7,
            negative_threshold=0.3,
            positive_fraction=0.5,
            samples_per_image=2,
        )
        rois = np.array(
            [
                [0, 0, 5, 5],
                [2.5, 2.5, 7.5, 7.5],
                [5, 5, 10, 10],
                [7.5, 7.5, 12.5, 12.5],
            ]
        )
        # the 3rd box will generate 0 IOUs and not sampled.
        gt_boxes = np.array([[10, 10, 15, 15], [2.5, 2.5, 7.5, 7.5]])
        gt_classes = np.array([2, 10, -1], dtype=np.int32)
        gt_classes = gt_classes[..., np.newaxis]
        box_targets, box_weights, cls_targets, cls_weights = rpn_encoder(
            rois, gt_boxes, gt_classes
        )
        # all rois will be matched to the 2nd gt boxes, and encoded
        expected_box_targets = (
            np.array(
                [
                    [0.5, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [-0.5, -0.5, 0.0, 0.0],
                    [0.5, 0.5, 0.0, 0.0],
                ]
            )
            / 0.1
        )
        self.assertAllClose(expected_box_targets, box_targets)
        # only foreground and background classes
        self.assertAllClose(np.max(ops.convert_to_numpy(cls_targets)), 1.0)
        self.assertAllClose(np.min(ops.convert_to_numpy(cls_targets)), 0.0)
        # all weights between 0 and 1
        self.assertAllClose(np.max(ops.convert_to_numpy(cls_weights)), 1.0)
        self.assertAllClose(np.min(ops.convert_to_numpy(cls_weights)), 0.0)
        self.assertAllClose(np.max(ops.convert_to_numpy(box_weights)), 1.0)
        self.assertAllClose(np.min(ops.convert_to_numpy(box_weights)), 0.0)

    def test_rpn_label_encoder_multi_level(self):
        self.skipTest(
            "TODO: resolving flaky test, https://github.com/keras-team/keras-cv/issues/2336"  # noqa
        )
        rpn_encoder = _RpnLabelEncoder(
            anchor_format="xyxy",
            ground_truth_box_format="xyxy",
            positive_threshold=0.7,
            negative_threshold=0.3,
            positive_fraction=0.5,
            samples_per_image=2,
        )
        rois = {
            2: np.array([[0, 0, 5, 5], [2.5, 2.5, 7.5, 7.5]]),
            3: np.array([[5, 5, 10, 10], [7.5, 7.5, 12.5, 12.5]]),
        }
        # the 3rd box will generate 0 IOUs and not sampled.
        gt_boxes = np.array([[10, 10, 15, 15], [2.5, 2.5, 7.5, 7.5]])
        gt_classes = np.array([2, 10, -1], dtype=np.float32)
        gt_classes = gt_classes[..., np.newaxis]
        _, _, _, cls_weights = rpn_encoder(rois, gt_boxes, gt_classes)
        # the 2nd level found 2 positive matches, the 3rd level found no match
        expected_cls_weights = {
            2: np.array([[0.0], [1.0]]),
            3: np.array([[0.0], [1.0]]),
        }
        self.assertAllClose(expected_cls_weights[2], cls_weights[2])
        self.assertAllClose(expected_cls_weights[3], cls_weights[3])

    def test_rpn_label_encoder_batched(self):
        rpn_encoder = _RpnLabelEncoder(
            anchor_format="xyxy",
            ground_truth_box_format="xyxy",
            positive_threshold=0.7,
            negative_threshold=0.3,
            positive_fraction=0.5,
            samples_per_image=2,
        )
        rois = np.array(
            [
                [0, 0, 5, 5],
                [2.5, 2.5, 7.5, 7.5],
                [5, 5, 10, 10],
                [7.5, 7.5, 12.5, 12.5],
            ]
        )
        # the 3rd box will generate 0 IOUs and not sampled.
        gt_boxes = np.array([[10, 10, 15, 15], [2.5, 2.5, 7.5, 7.5]])
        gt_classes = np.array([2, 10, -1], dtype=np.int32)
        gt_classes = gt_classes[..., np.newaxis]
        rois = rois[np.newaxis, ...]
        gt_boxes = gt_boxes[np.newaxis, ...]
        gt_classes = gt_classes[np.newaxis, ...]
        box_targets, box_weights, cls_targets, cls_weights = rpn_encoder(
            rois, gt_boxes, gt_classes
        )
        # all rois will be matched to the 2nd gt boxes, and encoded
        expected_box_targets = (
            np.array(
                [
                    [0.5, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [-0.5, -0.5, 0.0, 0.0],
                    [0.5, 0.5, 0.0, 0.0],
                ]
            )
            / 0.1
        )
        expected_box_targets = expected_box_targets[np.newaxis, ...]
        self.assertAllClose(expected_box_targets, box_targets)
        # only foreground and background classes
        self.assertAllClose(np.max(ops.convert_to_numpy(cls_targets)), 1.0)
        self.assertAllClose(np.min(ops.convert_to_numpy(cls_targets)), 0.0)
        # all weights between 0 and 1
        self.assertAllClose(np.max(ops.convert_to_numpy(cls_weights)), 1.0)
        self.assertAllClose(np.min(ops.convert_to_numpy(cls_weights)), 0.0)
        self.assertAllClose(np.max(ops.convert_to_numpy(box_weights)), 1.0)
        self.assertAllClose(np.min(ops.convert_to_numpy(box_weights)), 0.0)
