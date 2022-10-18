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

from keras_cv.layers.object_detection.rpn_label_encoder import _RpnLabelEncoder


class RpnLabelEncoderTest(tf.test.TestCase):
    def test_rpn_label_encoder(self):
        rpn_encoder = _RpnLabelEncoder(
            anchor_format="xyxy",
            ground_truth_box_format="xyxy",
            positive_threshold=0.7,
            negative_threshold=0.3,
            positive_fraction=0.5,
            samples_per_image=2,
        )
        rois = tf.constant(
            [[0, 0, 5, 5], [2.5, 2.5, 7.5, 7.5], [5, 5, 10, 10], [7.5, 7.5, 12.5, 12.5]]
        )
        # the 3rd box will generate 0 IOUs and not sampled.
        gt_boxes = tf.constant([[10, 10, 15, 15], [2.5, 2.5, 7.5, 7.5]])
        gt_classes = tf.constant([2, 10, -1], dtype=tf.int32)
        gt_classes = gt_classes[..., tf.newaxis]
        box_targets, box_weights, cls_targets, cls_weights = rpn_encoder(
            rois, gt_boxes, gt_classes
        )
        # all rois will be matched to the 2nd gt boxes, and encoded
        expected_box_targets = (
            tf.constant(
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
        self.assertAllClose(tf.reduce_max(cls_targets), 1.0)
        self.assertAllClose(tf.reduce_min(cls_targets), 0.0)
        # all weights between 0 and 1
        self.assertAllClose(tf.reduce_max(cls_weights), 1.0)
        self.assertAllClose(tf.reduce_min(cls_weights), 0.0)
        self.assertAllClose(tf.reduce_max(box_weights), 1.0)
        self.assertAllClose(tf.reduce_min(box_weights), 0.0)

    def test_rpn_label_encoder_multi_level(self):
        rpn_encoder = _RpnLabelEncoder(
            anchor_format="xyxy",
            ground_truth_box_format="xyxy",
            positive_threshold=0.7,
            negative_threshold=0.3,
            positive_fraction=0.5,
            samples_per_image=2,
        )
        rois = {
            2: tf.constant([[0, 0, 5, 5], [2.5, 2.5, 7.5, 7.5]]),
            3: tf.constant([[5, 5, 10, 10], [7.5, 7.5, 12.5, 12.5]]),
        }
        # the 3rd box will generate 0 IOUs and not sampled.
        gt_boxes = tf.constant([[10, 10, 15, 15], [2.5, 2.5, 7.5, 7.5]])
        gt_classes = tf.constant([2, 10, -1], dtype=tf.float32)
        gt_classes = gt_classes[..., tf.newaxis]
        _, _, _, cls_weights = rpn_encoder(rois, gt_boxes, gt_classes)
        # the 2nd level found 2 positive matches, the 3rd level found no match
        expected_cls_weights = {
            2: tf.constant([[0.0], [1.0]]),
            3: tf.constant([[0.0], [1.0]]),
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
        rois = tf.constant(
            [[0, 0, 5, 5], [2.5, 2.5, 7.5, 7.5], [5, 5, 10, 10], [7.5, 7.5, 12.5, 12.5]]
        )
        # the 3rd box will generate 0 IOUs and not sampled.
        gt_boxes = tf.constant([[10, 10, 15, 15], [2.5, 2.5, 7.5, 7.5]])
        gt_classes = tf.constant([2, 10, -1], dtype=tf.int32)
        gt_classes = gt_classes[..., tf.newaxis]
        rois = rois[tf.newaxis, ...]
        gt_boxes = gt_boxes[tf.newaxis, ...]
        gt_classes = gt_classes[tf.newaxis, ...]
        box_targets, box_weights, cls_targets, cls_weights = rpn_encoder(
            rois, gt_boxes, gt_classes
        )
        # all rois will be matched to the 2nd gt boxes, and encoded
        expected_box_targets = (
            tf.constant(
                [
                    [0.5, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [-0.5, -0.5, 0.0, 0.0],
                    [0.5, 0.5, 0.0, 0.0],
                ]
            )
            / 0.1
        )
        expected_box_targets = expected_box_targets[tf.newaxis, ...]
        self.assertAllClose(expected_box_targets, box_targets)
        # only foreground and background classes
        self.assertAllClose(tf.reduce_max(cls_targets), 1.0)
        self.assertAllClose(tf.reduce_min(cls_targets), 0.0)
        # all weights between 0 and 1
        self.assertAllClose(tf.reduce_max(cls_weights), 1.0)
        self.assertAllClose(tf.reduce_min(cls_weights), 0.0)
        self.assertAllClose(tf.reduce_max(box_weights), 1.0)
        self.assertAllClose(tf.reduce_min(box_weights), 0.0)
