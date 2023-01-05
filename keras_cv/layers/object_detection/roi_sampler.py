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

from keras_cv import bounding_box
from keras_cv.bounding_box import iou
from keras_cv.ops import box_matcher
from keras_cv.ops import sampling
from keras_cv.ops import target_gather


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class _ROISampler(tf.keras.layers.Layer):
    """
    Sample ROIs for loss related calucation.

    With proposals (ROIs) and ground truth, it performs the following:
    1) compute IOU similarity matrix
    2) match each proposal to ground truth box based on IOU
    3) samples positive matches and negative matches and return

    `append_gt_boxes` augments proposals with ground truth boxes. This is
    useful in 2 stage detection networks during initialization where the
    1st stage often cannot produce good proposals for 2nd stage. Setting it
    to True will allow it to generate more reasonable proposals at the begining.

    `background_class` allow users to set the labels for background proposals. Default
    is 0, where users need to manually shift the incoming `gt_classes` if its range is
    [0, num_classes).

    Args:
      bounding_box_format: The format of bounding boxes to generate. Refer
        [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
        for more details on supported bounding box formats.
      roi_matcher: a `ArgmaxBoxMatcher` object that matches proposals
        with ground truth boxes. the positive match must be 1 and negative match must be -1.
        Such assumption is not being validated here.
      positive_fraction: the positive ratio w.r.t `num_sampled_rois`. Defaults to 0.25.
      background_class: the background class which is used to map returned the sampled
        ground truth which is classified as background.
      num_sampled_rois: the number of sampled proposals per image for
        further (loss) calculation. Defaults to 256.
      append_gt_boxes: boolean, whether gt_boxes will be appended to rois
        before sample the rois. Defaults to True.
    """

    def __init__(
        self,
        bounding_box_format: str,
        roi_matcher: box_matcher.ArgmaxBoxMatcher,
        positive_fraction: float = 0.25,
        background_class: int = 0,
        num_sampled_rois: int = 256,
        append_gt_boxes: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format
        self.roi_matcher = roi_matcher
        self.positive_fraction = positive_fraction
        self.background_class = background_class
        self.num_sampled_rois = num_sampled_rois
        self.append_gt_boxes = append_gt_boxes
        self.built = True
        # for debugging.
        self._positives = tf.keras.metrics.Mean()
        self._negatives = tf.keras.metrics.Mean()

    def call(
        self,
        rois: tf.Tensor,
        gt_boxes: tf.Tensor,
        gt_classes: tf.Tensor,
    ):
        """
        Args:
          rois: [batch_size, num_rois, 4]
          gt_boxes: [batch_size, num_gt, 4]
          gt_classes: [batch_size, num_gt, 1]
        Returns:
          sampled_rois: [batch_size, num_sampled_rois, 4]
          sampled_gt_boxes: [batch_size, num_sampled_rois, 4]
          sampled_box_weights: [batch_size, num_sampled_rois, 1]
          sampled_gt_classes: [batch_size, num_sampled_rois, 1]
          sampled_class_weights: [batch_size, num_sampled_rois, 1]
        """
        if self.append_gt_boxes:
            # num_rois += num_gt
            rois = tf.concat([rois, gt_boxes], axis=1)
        num_rois = rois.get_shape().as_list()[1]
        if num_rois is None:
            raise ValueError(f"`rois` must have static shape, got {rois.get_shape()}")
        if num_rois < self.num_sampled_rois:
            raise ValueError(
                f"num_rois must be less than `num_sampled_rois` ({self.num_sampled_rois}), got {num_rois}"
            )
        rois = bounding_box.convert_format(
            rois, source=self.bounding_box_format, target="yxyx"
        )
        gt_boxes = bounding_box.convert_format(
            gt_boxes, source=self.bounding_box_format, target="yxyx"
        )
        # [batch_size, num_rois, num_gt]
        similarity_mat = iou.compute_iou(
            rois, gt_boxes, bounding_box_format="yxyx", use_masking=True
        )
        # [batch_size, num_rois] | [batch_size, num_rois]
        matched_gt_cols, matched_vals = self.roi_matcher(similarity_mat)
        # [batch_size, num_rois]
        positive_matches = tf.math.equal(matched_vals, 1)
        negative_matches = tf.math.equal(matched_vals, -1)
        self._positives.update_state(
            tf.reduce_sum(tf.cast(positive_matches, tf.float32), axis=-1)
        )
        self._negatives.update_state(
            tf.reduce_sum(tf.cast(negative_matches, tf.float32), axis=-1)
        )
        # [batch_size, num_rois, 1]
        background_mask = tf.expand_dims(tf.logical_not(positive_matches), axis=-1)
        # [batch_size, num_rois, 1]
        matched_gt_classes = target_gather._target_gather(gt_classes, matched_gt_cols)
        # also set all background matches to `background_class`
        matched_gt_classes = tf.where(
            background_mask,
            tf.cast(
                self.background_class * tf.ones_like(matched_gt_classes),
                gt_classes.dtype,
            ),
            matched_gt_classes,
        )
        # [batch_size, num_rois, 4]
        matched_gt_boxes = target_gather._target_gather(gt_boxes, matched_gt_cols)
        encoded_matched_gt_boxes = bounding_box._encode_box_to_deltas(
            anchors=rois,
            boxes=matched_gt_boxes,
            anchor_format="yxyx",
            box_format="yxyx",
            variance=[0.1, 0.1, 0.2, 0.2],
        )
        # also set all background matches to 0 coordinates
        encoded_matched_gt_boxes = tf.where(
            background_mask, tf.zeros_like(matched_gt_boxes), encoded_matched_gt_boxes
        )
        # [batch_size, num_rois]
        sampled_indicators = sampling.balanced_sample(
            positive_matches,
            negative_matches,
            self.num_sampled_rois,
            self.positive_fraction,
        )
        # [batch_size, num_sampled_rois] in the range of [0, num_rois)
        sampled_indicators, sampled_indices = tf.math.top_k(
            sampled_indicators, k=self.num_sampled_rois, sorted=True
        )
        # [batch_size, num_sampled_rois, 4]
        sampled_rois = target_gather._target_gather(rois, sampled_indices)
        # [batch_size, num_sampled_rois, 4]
        sampled_gt_boxes = target_gather._target_gather(
            encoded_matched_gt_boxes, sampled_indices
        )
        # [batch_size, num_sampled_rois, 1]
        sampled_gt_classes = target_gather._target_gather(
            matched_gt_classes, sampled_indices
        )
        # [batch_size, num_sampled_rois, 1]
        # all negative samples will be ignored in regression
        sampled_box_weights = target_gather._target_gather(
            tf.cast(positive_matches[..., tf.newaxis], gt_boxes.dtype), sampled_indices
        )
        # [batch_size, num_sampled_rois, 1]
        sampled_indicators = sampled_indicators[..., tf.newaxis]
        sampled_class_weights = tf.cast(sampled_indicators, gt_classes.dtype)
        return (
            sampled_rois,
            sampled_gt_boxes,
            sampled_box_weights,
            sampled_gt_classes,
            sampled_class_weights,
        )

    def get_config(self):
        config = {
            "bounding_box_format": self.bounding_box_format,
            "positive_fraction": self.positive_fraction,
            "background_class": self.background_class,
            "num_sampled_rois": self.num_sampled_rois,
            "append_gt_boxes": self.append_gt_boxes,
            "roi_matcher": self.roi_matcher.get_config(),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        roi_matcher_config = config.pop("roi_matcher")
        roi_matcher = box_matcher.ArgmaxBoxMatcher(**roi_matcher_config)
        return cls(roi_matcher=roi_matcher, **config)
