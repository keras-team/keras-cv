# Copyright 2023 The KerasCV Authors
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
"""Label encoder for YOLOV8. This uses the TOOD Task Aligned Assigner approach,
and is adapted from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/utils/tal.py
"""  # noqa: E501

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras_cv import bounding_box
from keras_cv.bounding_box.iou import compute_ciou


def select_highest_overlaps(mask_pos, overlaps, max_num_boxes):
    """Break ties when two GT boxes match to the same anchor.

    Picks the GT box with the highest IoU.
    """
    # (b, max_num_boxes, num_anchors) -> (b, num_anchors)
    fg_mask = tf.reduce_sum(mask_pos, axis=-2)

    def handle_anchor_with_two_gt_boxes(
        fg_mask, mask_pos, overlaps, max_num_boxes
    ):
        mask_multi_gts = tf.repeat(
            tf.expand_dims(fg_mask, axis=1) > 1, max_num_boxes, axis=1
        )  # (b, max_num_boxes, num_anchors)
        max_overlaps_idx = tf.argmax(overlaps, axis=1)  # (b, num_anchors)
        is_max_overlaps = tf.one_hot(
            max_overlaps_idx,
            tf.cast(max_num_boxes, dtype=tf.int32),  # tf.one_hot must use int32
        )  # (b, num_anchors, max_num_boxes)
        is_max_overlaps = tf.cast(
            tf.transpose(is_max_overlaps, perm=(0, 2, 1)), overlaps.dtype
        )  # (b, max_num_boxes, num_anchors)
        mask_pos = tf.where(
            mask_multi_gts, is_max_overlaps, mask_pos
        )  # (b, max_num_boxes, num_anchors)
        fg_mask = tf.reduce_sum(mask_pos, axis=-2)
        return fg_mask, mask_pos

    fg_mask, mask_pos = tf.cond(
        tf.reduce_max(fg_mask) > 1,
        lambda: handle_anchor_with_two_gt_boxes(
            fg_mask, mask_pos, overlaps, max_num_boxes
        ),
        lambda: (fg_mask, mask_pos),
    )

    target_gt_idx = tf.argmax(mask_pos, axis=-2)  # (b, num_anchors)
    return target_gt_idx, fg_mask, mask_pos


def select_candidates_in_gts(xy_centers, gt_bboxes, epsilon=1e-9):
    """Selects candidate anchors for GT boxes.

    Returns:
        a boolean mask Tensor of shape (batch_size, num_gt_boxes, num_anchors)
        where the value is `True` if the anchor point falls inside the gt box,
        and `False` otherwise.
    """
    n_anchors = xy_centers.shape[0]
    n_boxes = tf.shape(gt_bboxes)[1]

    left_top, right_bottom = tf.split(
        tf.reshape(gt_bboxes, (-1, 1, 4)), 2, axis=-1
    )
    bbox_deltas = tf.reshape(
        tf.concat(
            [
                xy_centers[tf.newaxis] - left_top,
                right_bottom - xy_centers[tf.newaxis],
            ],
            axis=2,
        ),
        (-1, n_boxes, n_anchors, 4),
    )

    return tf.reduce_min(bbox_deltas, axis=-1) > epsilon


@keras.utils.register_keras_serializable(package="keras_cv")
class YOLOV8LabelEncoder(layers.Layer):
    """
    Encodes ground truth boxes to target boxes and class labels for training a
    YOLOV8 model. This is an implementation of the Task-aligned sample
    assignment scheme proposed in https://arxiv.org/abs/2108.07755.

    Args:
        num_classes: integer, the number of classes in the training dataset
        max_anchor_matches: optional integer, the maximum number of anchors to
            match with any given ground truth box. For example, when the default
            10 is used, the 10 candidate anchor points with the highest
            alignment score are matched with a ground truth box. If less than 10
            candidate anchors exist, all candidates will be matched to the box.
        alpha: float, a parameter to control the influence of class predictions
            on the alignment score of an anchor box. This is the alpha parameter
            in equation 9 of https://arxiv.org/pdf/2108.07755.pdf.
        beta: float, a parameter to control the influence of box IOUs on the
            alignment score of an anchor box. This is the beta parameter in
            equation 9 of https://arxiv.org/pdf/2108.07755.pdf.
        epsilon: float, a small number used for numerical stability in division
            (to avoid diving by zero), and used as a threshold to eliminate very
            small matches based on alignment scores of approximately zero.
    """

    def __init__(
        self,
        num_classes,
        max_anchor_matches=10,
        alpha=0.5,
        beta=6.0,
        epsilon=1e-9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_anchor_matches = max_anchor_matches
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def call(
        self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt
    ):
        """Computes target boxes and classes for anchors.

        Args:
            pd_scores: a Float Tensor of shape (batch_size, num_anchors,
                num_classes) representing predicted class scores for each
                anchor.
            pd_bboxes: a Float Tensor of shape (batch_size, num_anchors, 4)
                representing predicted boxes for each anchor.
            anc_points: a Float Tensor of shape (batch_size, num_anchors, 2)
                representing the xy coordinates of the center of each anchor.
            gt_labels: a Float Tensor of shape (batch_size, num_gt_boxes)
                representing the classes of ground truth boxes.
            gt_bboxes: a Float Tensor of shape (batch_size, num_gt_boxes, 4)
                representing the ground truth bounding boxes in xyxy format.
            mask_gt: A Boolean Tensor of shape (batch_size, num_gt_boxes)
                representing whether a box in `gt_bboxes` is a real box or a
                non-box that exists due to padding.

        Returns:
            A tuple of the following:
                - A Float Tensor of shape (batch_size, num_anchors, 4)
                    representing box targets for the model.
                - A Float Tensor of shape (batch_size, num_anchors, num_classes)
                    representing class targets for the model.
                - A Boolean Tensor of shape (batch_size, num_anchors)
                    representing whether each anchor was a match with a ground
                    truth box. Anchors that didn't match with a ground truth
                    box should be excluded from both class and box losses.
        """
        if isinstance(gt_bboxes, tf.RaggedTensor):
            dense_bounding_boxes = bounding_box.to_dense(
                {"boxes": gt_bboxes, "classes": gt_labels},
            )
            gt_bboxes = dense_bounding_boxes["boxes"]
            gt_labels = dense_bounding_boxes["classes"]

        if isinstance(mask_gt, tf.RaggedTensor):
            mask_gt = mask_gt.to_tensor()

        max_num_boxes = tf.cast(tf.shape(gt_bboxes)[1], dtype=tf.int64)

        def handle_call(
            pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt
        ):
            mask_pos, align_metric, overlaps = self.get_pos_mask(
                pd_scores,
                pd_bboxes,
                gt_labels,
                gt_bboxes,
                anc_points,
                mask_gt,
                max_num_boxes,
            )

            target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(
                mask_pos, overlaps, max_num_boxes
            )

            target_bboxes, target_scores = self.get_targets(
                gt_labels, gt_bboxes, target_gt_idx, fg_mask, max_num_boxes
            )

            align_metric *= mask_pos
            pos_align_metrics = tf.reduce_max(
                align_metric, axis=-1, keepdims=True
            )  # b, max_num_boxes
            pos_overlaps = tf.reduce_max(
                overlaps * mask_pos, axis=-1, keepdims=True
            )  # b, max_num_boxes
            norm_align_metric = tf.expand_dims(
                tf.reduce_max(
                    align_metric
                    * pos_overlaps
                    / (pos_align_metrics + self.epsilon),
                    axis=-2,
                ),
                axis=-1,
            )
            target_scores = target_scores * norm_align_metric

            # No need to compute gradients for these, as they're all targets
            return (
                tf.stop_gradient(target_bboxes),
                tf.stop_gradient(target_scores),
                tf.stop_gradient(tf.cast(fg_mask, tf.bool)),
            )

        # return zeros if no gt boxes are present
        return tf.cond(
            max_num_boxes > 0,
            lambda: handle_call(
                pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt
            ),
            lambda: (
                tf.zeros_like(pd_bboxes),
                tf.zeros_like(pd_scores),
                tf.zeros_like(pd_scores[..., 0], dtype=tf.bool),
            ),
        )

    def get_pos_mask(
        self,
        pd_scores,
        pd_bboxes,
        gt_labels,
        gt_bboxes,
        anc_points,
        mask_gt,
        max_num_boxes,
    ):
        """Identifies matches between gt boxes and anchors.

        Returns:
            A tuple of the following:
                - A Boolean Tensor of shape (batch_size, num_gt_boxes,
                    num_anchors) representing whether each gt box has matched
                    with each anchor.
                - A Float Tensor of shape (batch_size, num_gt_boxes,
                    num_anchors) representing the alignment score of each gt box
                    with each anchor.
                - A Float Tensor or shape (batch_size, num_gt_boxes,
                    num_anchors representing the IoU of each GT box with the
                    predicted box at each anchor.
        """
        # get in_gts mask, (b, max_num_boxes, num_anchors)
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)

        align_metric, overlaps = self.get_box_metrics(
            pd_scores,
            pd_bboxes,
            gt_labels,
            gt_bboxes,
            tf.cast(mask_in_gts, tf.int32) * tf.cast(mask_gt, tf.int32),
            max_num_boxes,
        )
        # get topk_metric mask, (b, max_num_boxes, num_anchors)
        mask_topk = self.select_topk_candidates(
            align_metric,
            topk_mask=tf.cast(
                tf.repeat(mask_gt, self.max_anchor_matches, axis=2), tf.bool
            ),
        )
        # merge all masks to a final mask, (b, max_num_boxes, num_anchors)
        mask_pos = (
            mask_topk
            * tf.cast(mask_in_gts, tf.float32)
            * tf.cast(mask_gt, tf.float32)
        )

        return mask_pos, align_metric, overlaps

    def get_box_metrics(
        self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt, max_num_boxes
    ):
        """Computes alignment metrics for each gt box, anchor pair.

        Returns:
            A tuple of the following:
                - A Float Tensor of shape (batch_size, num_gt_boxes,
                    num_anchors) representing the alignment metrics for each
                    ground truth box, anchor pair.
                - A Float Tensor of shape (batch_size, num_gt_boxes,
                    num_anchors) representing the IoUs between each ground truth
                    box and the predicted box at each anchor.
        """
        na = pd_bboxes.shape[-2]
        mask_gt = tf.cast(mask_gt, tf.bool)  # b, max_num_boxes, num_anchors

        ind_1 = tf.cast(gt_labels, tf.int64)
        pd_scores = tf.gather(
            pd_scores, tf.math.maximum(ind_1, 0), axis=-1, batch_dims=1
        )
        pd_scores = tf.where(ind_1[:, tf.newaxis, :] >= 0, pd_scores, 0.0)
        pd_scores = tf.transpose(pd_scores, perm=(0, 2, 1))

        bbox_scores = tf.where(mask_gt, pd_scores, 0.0)

        pd_boxes = tf.repeat(
            tf.expand_dims(pd_bboxes, axis=1), max_num_boxes, axis=1
        )

        gt_boxes = tf.repeat(tf.expand_dims(gt_bboxes, axis=2), na, axis=2)

        iou = tf.squeeze(
            compute_ciou(gt_boxes, pd_boxes, bounding_box_format="xyxy"),
            axis=-1,
        )
        iou = tf.where(iou > 0, iou, 0.0)

        iou = tf.reshape(iou, (-1, max_num_boxes, na))
        overlaps = tf.where(mask_gt, iou, 0.0)

        align_metric = tf.math.pow(bbox_scores, self.alpha) * tf.math.pow(
            overlaps, self.beta
        )
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, topk_mask):
        """Selects the anchors with the top-k alignment metrics for each gt box.

        Returns:
            A Boolean Tensor of shape (batch_size, num_gt_boxes, num_anchors)
            representing whether each anchor is among the top-k anchors for a
            given gt box based on the alignment metric.
        """

        num_anchors = metrics.shape[-1]  # num_anchors
        # (b, max_num_boxes, topk)
        topk_metrics, topk_idxs = tf.math.top_k(
            metrics, self.max_anchor_matches
        )
        topk_mask = tf.tile(
            tf.reduce_max(topk_metrics, axis=-1, keepdims=True) > self.epsilon,
            [1, 1, self.max_anchor_matches],
        )

        # (b, max_num_boxes, topk)
        topk_idxs = tf.where(topk_mask, topk_idxs, 0)
        is_in_topk = tf.zeros_like(metrics, dtype=tf.int64)

        for it in range(self.max_anchor_matches):
            is_in_topk += tf.one_hot(
                topk_idxs[:, :, it], num_anchors, dtype=tf.int64
            )

        # filter invalid bboxes
        is_in_topk = tf.where(
            is_in_topk > 1, tf.constant(0, tf.int64), is_in_topk
        )
        return tf.cast(is_in_topk, metrics.dtype)

    def get_targets(
        self, gt_labels, gt_bboxes, target_gt_idx, fg_mask, max_num_boxes
    ):
        """Computes target boxes and labels.

        Returns:
            A tuple of the following:
                - A Float Tensor of shape (batch_size, num_anchors, 4)
                    representing target boxes each anchor..
                - A Float Tensor of shape (batch_size, num_anchors, num_classes)
                    representing target classes for each anchor.
        """

        batch_ind = tf.range(tf.shape(gt_labels)[0], dtype=tf.int64)[
            ..., tf.newaxis
        ]
        target_gt_idx = target_gt_idx + batch_ind * max_num_boxes

        gt_bboxes = tf.reshape(gt_bboxes, (-1, tf.shape(gt_bboxes)[1], 4))

        target_labels = tf.gather(
            tf.reshape(tf.cast(gt_labels, tf.int64), (-1,)), target_gt_idx
        )  # (b, num_anchors)

        # assigned target boxes, (b, max_num_boxes, 4) -> (b, num_anchors)
        target_bboxes = tf.gather(
            tf.reshape(gt_bboxes, (-1, 4)), target_gt_idx, axis=-2
        )

        # assigned target scores
        target_labels = tf.math.maximum(target_labels, 0)
        target_scores = tf.one_hot(
            target_labels, self.num_classes
        )  # (b, num_anchors, num_classes)
        fg_scores_mask = tf.repeat(
            fg_mask[:, :, tf.newaxis], self.num_classes, axis=2
        )  # (b, num_anchors, num_classes)
        target_scores = tf.where(fg_scores_mask > 0, target_scores, 0)

        return target_bboxes, target_scores

    def count_params(self):
        # The label encoder has no weights, so we short-circuit the weight
        # counting to avoid having to `build` this layer unnecessarily.
        return 0

    def get_config(self):
        config = {
            "max_anchor_matches": self.max_anchor_matches,
            "num_classes": self.num_classes,
            "alpha": self.alpha,
            "beta": self.beta,
            "epsilon": self.epsilon,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
