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

from keras_cv.layers.object_detection.roi_pool import ROIPooler


class ROIPoolTest(tf.test.TestCase):
    def test_no_quantize(self):
        roi_pooler = ROIPooler(
            "rel_yxyx", target_size=[2, 2], image_shape=[224, 224, 3]
        )
        feature_map = tf.expand_dims(tf.reshape(tf.range(64), [8, 8, 1]), axis=0)
        rois = tf.reshape(tf.constant([0.0, 0.0, 1.0, 1.0]), [1, 1, 4])
        pooled_feature_map = roi_pooler(feature_map, rois)
        # the maximum value would be at bottom-right at each block, roi sharded into 2x2 blocks
        # | 0, 1, 2, 3          | 4, 5, 6, 7            |
        # | 8, 9, 10, 11        | 12, 13, 14, 15        |
        # | 16, 17, 18, 19      | 20, 21, 22, 23        |
        # | 24, 25, 26, 27(max) | 28, 29, 30, 31(max)   |
        # --------------------------------------------
        # | 32, 33, 34, 35      | 36, 37, 38, 39        |
        # | 40, 41, 42, 43      | 44, 45, 46, 47        |
        # | 48, 49, 50, 51      | 52, 53, 54, 55        |
        # | 56, 57, 58, 59(max) | 60, 61, 62, 63(max)   |
        # --------------------------------------------
        expected_feature_map = tf.reshape(tf.constant([27, 31, 59, 63]), [1, 2, 2, 1])
        self.assertAllClose(expected_feature_map, pooled_feature_map)

    def test_roi_quantize_y(self):
        roi_pooler = ROIPooler("yxyx", target_size=[2, 2], image_shape=[224, 224, 3])
        feature_map = tf.expand_dims(tf.reshape(tf.range(64), [8, 8, 1]), axis=0)
        rois = tf.reshape(tf.constant([0.0, 0.0, 224, 220]), [1, 1, 4])
        pooled_feature_map = roi_pooler(feature_map, rois)
        # the maximum value would be at bottom-right at each block, roi sharded into 2x2 blocks
        # | 0, 1, 2             | 3, 4, 5, 6            | 7 (removed)
        # | 8, 9, 10            | 11, 12, 13, 14        | 15 (removed)
        # | 16, 17, 18          | 19, 20, 21, 22        | 23 (removed)
        # | 24, 25, 26(max)     | 27, 28, 29, 30(max)   | 31 (removed)
        # --------------------------------------------
        # | 32, 33, 34          | 35, 36, 37, 38        | 39 (removed)
        # | 40, 41, 42          | 43, 44, 45, 46        | 47 (removed)
        # | 48, 49, 50          | 51, 52, 53, 54        | 55 (removed)
        # | 56, 57, 58(max)     | 59, 60, 61, 62(max)   | 63 (removed)
        # --------------------------------------------
        expected_feature_map = tf.reshape(tf.constant([26, 30, 58, 62]), [1, 2, 2, 1])
        self.assertAllClose(expected_feature_map, pooled_feature_map)

    def test_roi_quantize_x(self):
        roi_pooler = ROIPooler("yxyx", target_size=[2, 2], image_shape=[224, 224, 3])
        feature_map = tf.expand_dims(tf.reshape(tf.range(64), [8, 8, 1]), axis=0)
        rois = tf.reshape(tf.constant([0.0, 0.0, 220, 224]), [1, 1, 4])
        pooled_feature_map = roi_pooler(feature_map, rois)
        # the maximum value would be at bottom-right at each block, roi sharded into 2x2 blocks
        # | 0, 1, 2, 3          | 4, 5, 6, 7            |
        # | 8, 9, 10, 11        | 12, 13, 14, 15        |
        # | 16, 17, 18, 19(max) | 20, 21, 22, 23(max)   |
        # --------------------------------------------
        # | 24, 25, 26, 27      | 28, 29, 30, 31        |
        # | 32, 33, 34, 35      | 36, 37, 38, 39        |
        # | 40, 41, 42, 43      | 44, 45, 46, 47        |
        # | 48, 49, 50, 51(max) | 52, 53, 54, 55(max)   |
        # --------------------------------------------
        expected_feature_map = tf.reshape(tf.constant([19, 23, 51, 55]), [1, 2, 2, 1])
        self.assertAllClose(expected_feature_map, pooled_feature_map)

    def test_roi_quantize_h(self):
        roi_pooler = ROIPooler("yxyx", target_size=[3, 2], image_shape=[224, 224, 3])
        feature_map = tf.expand_dims(tf.reshape(tf.range(64), [8, 8, 1]), axis=0)
        rois = tf.reshape(tf.constant([0.0, 0.0, 224, 224]), [1, 1, 4])
        pooled_feature_map = roi_pooler(feature_map, rois)
        # the maximum value would be at bottom-right at each block, roi sharded into 3x2 blocks
        # | 0, 1, 2, 3          | 4, 5, 6, 7            |
        # | 8, 9, 10, 11(max)   | 12, 13, 14, 15(max)   |
        # --------------------------------------------
        # | 16, 17, 18, 19      | 20, 21, 22, 23        |
        # | 24, 25, 26, 27      | 28, 29, 30, 31        |
        # | 32, 33, 34, 35(max) | 36, 37, 38, 39(max)   |
        # --------------------------------------------
        # | 40, 41, 42, 43      | 44, 45, 46, 47        |
        # | 48, 49, 50, 51      | 52, 53, 54, 55        |
        # | 56, 57, 58, 59(max) | 60, 61, 62, 63(max)   |
        # --------------------------------------------
        expected_feature_map = tf.reshape(
            tf.constant([11, 15, 35, 39, 59, 63]), [1, 3, 2, 1]
        )
        self.assertAllClose(expected_feature_map, pooled_feature_map)

    def test_roi_quantize_w(self):
        roi_pooler = ROIPooler("yxyx", target_size=[2, 3], image_shape=[224, 224, 3])
        feature_map = tf.expand_dims(tf.reshape(tf.range(64), [8, 8, 1]), axis=0)
        rois = tf.reshape(tf.constant([0.0, 0.0, 224, 224]), [1, 1, 4])
        pooled_feature_map = roi_pooler(feature_map, rois)
        # the maximum value would be at bottom-right at each block, roi sharded into 2x3 blocks
        # | 0, 1        | 2, 3, 4           | 5, 6, 7           |
        # | 8, 9        | 10, 11, 12        | 13, 14, 15        |
        # | 16, 17      | 18, 19, 20        | 21, 22, 23        |
        # | 24, 25(max) | 26, 27, 28(max)   | 29, 30, 31(max)   |
        # --------------------------------------------
        # | 32, 33      | 34, 35, 36        | 37, 38, 39        |
        # | 40, 41      | 42, 43, 44        | 45, 46, 47        |
        # | 48, 49      | 50, 51, 52        | 53, 54, 55        |
        # | 56, 57(max) | 58, 59, 60(max)   | 61, 62, 63(max)   |
        # --------------------------------------------
        expected_feature_map = tf.reshape(
            tf.constant([25, 28, 31, 57, 60, 63]), [1, 2, 3, 1]
        )
        self.assertAllClose(expected_feature_map, pooled_feature_map)

    def test_roi_feature_map_height_smaller_than_roi(self):
        roi_pooler = ROIPooler("yxyx", target_size=[6, 2], image_shape=[224, 224, 3])
        feature_map = tf.expand_dims(tf.reshape(tf.range(16), [4, 4, 1]), axis=0)
        rois = tf.reshape(tf.constant([0.0, 0.0, 224, 224]), [1, 1, 4])
        pooled_feature_map = roi_pooler(feature_map, rois)
        # | 0, 1(max)   | 2, 3(max)     |
        # ------------------repeated----------------------
        # | 4, 5(max)   | 6, 7(max)     |
        # --------------------------------------------
        # | 8, 9(max)   | 10, 11(max)   |
        # ------------------repeated----------------------
        # | 12, 13(max) | 14, 15(max)   |
        expected_feature_map = tf.reshape(
            tf.constant([1, 3, 1, 3, 5, 7, 9, 11, 9, 11, 13, 15]), [1, 6, 2, 1]
        )
        self.assertAllClose(expected_feature_map, pooled_feature_map)

    def test_roi_feature_map_width_smaller_than_roi(self):
        roi_pooler = ROIPooler("yxyx", target_size=[2, 6], image_shape=[224, 224, 3])
        feature_map = tf.expand_dims(tf.reshape(tf.range(16), [4, 4, 1]), axis=0)
        rois = tf.reshape(tf.constant([0.0, 0.0, 224, 224]), [1, 1, 4])
        pooled_feature_map = roi_pooler(feature_map, rois)
        # | 0       | 1         | 2         | 3         |
        # | 4(max)  | 5(max)    | 6(max)    | 7(max)    |
        # --------------------------------------------
        # | 8       | 9         | 10        | 11        |
        # | 12(max) | 13(max)   | 14(max)   | 15(max)   |
        # --------------------------------------------
        expected_feature_map = tf.reshape(
            tf.constant([4, 4, 5, 6, 6, 7, 12, 12, 13, 14, 14, 15]), [1, 2, 6, 1]
        )
        self.assertAllClose(expected_feature_map, pooled_feature_map)

    def test_roi_empty(self):
        roi_pooler = ROIPooler("yxyx", target_size=[2, 2], image_shape=[224, 224, 3])
        feature_map = tf.expand_dims(tf.reshape(tf.range(1, 65), [8, 8, 1]), axis=0)
        rois = tf.reshape(tf.constant([0.0, 0.0, 0.0, 0.0]), [1, 1, 4])
        pooled_feature_map = roi_pooler(feature_map, rois)
        # all outputs should be top-left pixel
        self.assertAllClose(tf.ones([1, 2, 2, 1]), pooled_feature_map)

    def test_invalid_image_shape(self):
        with self.assertRaisesRegex(ValueError, "dynamic shape"):
            _ = ROIPooler("rel_yxyx", target_size=[2, 2], image_shape=[None, 224, 3])
