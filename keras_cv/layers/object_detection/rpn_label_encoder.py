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

from typing import Mapping

import tensorflow as tf

from keras_cv import bounding_box
from keras_cv.bounding_box import iou
from keras_cv.ops import box_matcher
from keras_cv.ops import sampling
from keras_cv.ops import target_gather


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class _RpnLabelEncoder(tf.keras.layers.Layer):
    """Transforms the raw labels into training targets for region proposal network (RPN).

    # TODO(tanzhenyu): consider unifying with _ROISampler.
    This is different from _ROISampler for a couple of reasons:
    1) This deals with unbatched input, dict of anchors and potentially ragged labels
    2) This deals with ground truth boxes, while _ROISampler deals with padded ground truth
       boxes with value -1 and padded ground truth classes with value -1
    3) this returns positive class target as 1, while _ROISampler returns
       positive class target as-is. (All negative class target are 0)
       The final classification loss will use one hot and #num_fg_classes + 1
    4) this returns #num_anchors dense targets, while _ROISampler returns
       #num_sampled_rois dense targets.
    5) this returns all positive box targets, while _ROISampler still samples
       positive box targets, while all negative box targets are also ignored
       in regression loss.

    Args:
      anchor_format: The format of bounding boxes for anchors to generate. Refer
        [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
        for more details on supported bounding box formats.
      ground_truth_box_format: The format of bounding boxes for ground truth boxes to generate.
      positive_threshold: the float threshold to set an anchor to positive match to gt box.
        values above it are positive matches.
      negative_threshold: the float threshold to set an anchor to negative match to gt box.
        values below it are negative matches.
      samples_per_image: for each image, the number of positive and negative samples
        to generate.
      positive_fraction: the fraction of positive samples to the total samples.

    """

    def __init__(
        self,
        anchor_format,
        ground_truth_box_format,
        positive_threshold,
        negative_threshold,
        samples_per_image,
        positive_fraction,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.anchor_format = anchor_format
        self.ground_truth_box_format = ground_truth_box_format
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.samples_per_image = samples_per_image
        self.positive_fraction = positive_fraction
        self.box_matcher = box_matcher.ArgmaxBoxMatcher(
            thresholds=[negative_threshold, positive_threshold],
            match_values=[-1, -2, 1],
            force_match_for_each_col=False,
        )
        self.box_variance = box_variance
        self.built = True
        self._positives = tf.keras.metrics.Mean()

    def call(
        self,
        anchors_dict: Mapping[str, tf.Tensor],
        gt_boxes: tf.Tensor,
        gt_classes: tf.Tensor,
    ):
        """
        Args:
          anchors: dict of [num_anchors, 4] or [batch_size, num_anchors, 4]
            float Tensor for each level.
          gt_boxes: [num_gt, 4] or [batch_size, num_anchors] float Tensor.
          gt_classes: [num_gt, 1] float or integer Tensor.
        Returns:
          box_targets: dict of [num_anchors, 4] or  for each level.
          box_weights: dict of [num_anchors, 1] for each level.
          class_targets: dict of [num_anchors, 1] for each level.
          class_weights: dict of [num_anchors, 1] for each level.
        """
        pack = False
        anchors = anchors_dict
        if isinstance(anchors, dict):
            pack = True
            anchors = tf.concat(tf.nest.flatten(anchors), axis=0)
        anchors = bounding_box.convert_format(
            anchors, source=self.anchor_format, target="yxyx"
        )
        gt_boxes = bounding_box.convert_format(
            gt_boxes, source=self.ground_truth_box_format, target="yxyx"
        )
        # [num_anchors, num_gt] or [batch_size, num_anchors, num_gt]
        similarity_mat = iou.compute_iou(anchors, gt_boxes, bounding_box_format="yxyx")
        # [num_anchors] or [batch_size, num_anchors]
        matched_gt_indices, matched_vals = self.box_matcher(similarity_mat)
        # [num_anchors] or [batch_size, num_anchors]
        positive_matches = tf.math.equal(matched_vals, 1)
        # currently SyncOnReadVariable does not support `assign_add` in cross-replica.
        #        self._positives.update_state(
        #            tf.reduce_sum(tf.cast(positive_matches, tf.float32), axis=-1)
        #        )

        negative_matches = tf.math.equal(matched_vals, -1)
        # [num_anchors, 4] or [batch_size, num_anchors, 4]
        matched_gt_boxes = target_gather._target_gather(gt_boxes, matched_gt_indices)
        # [num_anchors, 4] or [batch_size, num_anchors, 4], used as `y_true` for regression loss
        encoded_box_targets = bounding_box._encode_box_to_deltas(
            anchors,
            matched_gt_boxes,
            anchor_format="yxyx",
            box_format="yxyx",
            variance=self.box_variance,
        )
        # [num_anchors, 1] or [batch_size, num_anchors, 1]
        box_sample_weights = tf.cast(positive_matches[..., tf.newaxis], gt_boxes.dtype)

        # [num_anchors, 1] or [batch_size, num_anchors, 1]
        positive_mask = tf.expand_dims(positive_matches, axis=-1)
        # set all negative and ignored matches to 0, and all positive matches to 1
        # [num_anchors, 1] or [batch_size, num_anchors, 1]
        positive_classes = tf.ones_like(positive_mask, dtype=gt_classes.dtype)
        negative_classes = tf.zeros_like(positive_mask, dtype=gt_classes.dtype)
        # [num_anchors, 1] or [batch_size, num_anchors, 1]
        class_targets = tf.where(positive_mask, positive_classes, negative_classes)
        # [num_anchors] or [batch_size, num_anchors]
        sampled_indicators = sampling.balanced_sample(
            positive_matches,
            negative_matches,
            self.samples_per_image,
            self.positive_fraction,
        )
        # [num_anchors, 1] or [batch_size, num_anchors, 1]
        class_sample_weights = tf.cast(
            sampled_indicators[..., tf.newaxis], gt_classes.dtype
        )
        if pack:
            encoded_box_targets = self.unpack_targets(encoded_box_targets, anchors_dict)
            box_sample_weights = self.unpack_targets(box_sample_weights, anchors_dict)
            class_targets = self.unpack_targets(class_targets, anchors_dict)
            class_sample_weights = self.unpack_targets(
                class_sample_weights, anchors_dict
            )
        return (
            encoded_box_targets,
            box_sample_weights,
            class_targets,
            class_sample_weights,
        )

    def unpack_targets(self, targets, anchors_dict):
        target_shape = len(targets.get_shape().as_list())
        if target_shape != 2 and target_shape != 3:
            raise ValueError(
                f"unpacking targets must be rank 2 or rank 3, got {target_shape}"
            )
        unpacked_targets = {}
        count = 0
        for level, anchors in anchors_dict.items():
            num_anchors_lvl = anchors.get_shape().as_list()[0]
            if target_shape == 2:
                unpacked_targets[level] = targets[count : count + num_anchors_lvl, ...]
            else:
                unpacked_targets[level] = targets[
                    :, count : count + num_anchors_lvl, ...
                ]
            count += num_anchors_lvl
        return unpacked_targets

    def get_config(self):
        config = {
            "anchor_format": self.anchor_format,
            "ground_truth_box_format": self.ground_truth_box_format,
            "positive_threshold": self.positive_threshold,
            "negative_threshold": self.negative_threshold,
            "samples_per_image": self.samples_per_image,
            "positive_fraction": self.positive_fraction,
            "box_variance": self.box_variance,
        }
        return config
