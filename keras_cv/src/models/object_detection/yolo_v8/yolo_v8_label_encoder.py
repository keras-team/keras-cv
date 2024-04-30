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
"""Label encoder for YOLOV8. This uses the TOOD Task Aligned Assigner approach.
See https://arxiv.org/abs/2108.07755 for more info, as well as a reference
implementation at https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/task_aligned_assigner.py
"""  # noqa: E501

import tensorflow as tf

from keras_cv.src import bounding_box
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.bounding_box.iou import compute_ciou


def is_anchor_center_within_box(anchors, gt_bboxes):
    return ops.all(
        ops.logical_and(
            gt_bboxes[:, :, None, :2] < anchors,
            gt_bboxes[:, :, None, 2:] > anchors,
        ),
        axis=-1,
    )


@keras_cv_export("keras_cv.models.yolov8.LabelEncoder")
class YOLOV8LabelEncoder(keras.layers.Layer):
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

    def assign(
        self, scores, decode_bboxes, anchors, gt_labels, gt_bboxes, gt_mask
    ):
        """Assigns ground-truth boxes to anchors.

        Uses the task-aligned assignment strategy for matching ground truth
        and anchor boxes based on prediction scores and IoU.
        """
        num_anchors = anchors.shape[0]

        # Box scores are the predicted scores for each anchor, ground truth box
        # pair. Only the predicted score for the class of the GT box is included
        # Shape: (B, num_gt_boxes, num_anchors) (after transpose)
        bbox_scores = ops.take_along_axis(
            scores,
            ops.cast(ops.maximum(gt_labels[:, None, :], 0), "int32"),
            axis=-1,
        )
        bbox_scores = ops.transpose(bbox_scores, (0, 2, 1))

        # Overlaps are the IoUs of each predicted box and each GT box.
        # Shape: (B, num_gt_boxes, num_anchors)
        overlaps = compute_ciou(
            ops.expand_dims(gt_bboxes, axis=2),
            ops.expand_dims(decode_bboxes, axis=1),
            bounding_box_format="xyxy",
        )

        # Alignment metrics are a combination of box scores and overlaps, per
        # the task-aligned-assignment formula.
        # Metrics are forced to 0 for boxes which have been masked in the GT
        # input (e.g. due to padding)
        alignment_metrics = ops.power(bbox_scores, self.alpha) * ops.power(
            overlaps, self.beta
        )
        alignment_metrics = ops.where(gt_mask, alignment_metrics, 0)

        # Only anchors which are inside of relevant GT boxes are considered
        # for assignment.
        # This is a boolean tensor of shape (B, num_gt_boxes, num_anchors)
        matching_anchors_in_gt_boxes = is_anchor_center_within_box(
            anchors, gt_bboxes
        )
        alignment_metrics = ops.where(
            matching_anchors_in_gt_boxes, alignment_metrics, 0
        )

        # The top-k highest alignment metrics are used to select K candidate
        # anchors for each GT box.
        candidate_metrics, candidate_idxs = ops.top_k(
            alignment_metrics, self.max_anchor_matches
        )
        candidate_idxs = ops.where(candidate_metrics > 0, candidate_idxs, -1)

        # We now compute a dense grid of anchors and GT boxes. This is useful
        # for picking a GT box when an anchor matches to 2, as well as returning
        # to a dense format for a mask of which anchors have been matched.
        anchors_matched_gt_box = ops.zeros_like(overlaps)
        for k in range(self.max_anchor_matches):
            anchors_matched_gt_box += ops.one_hot(
                candidate_idxs[:, :, k], num_anchors
            )

        # We zero-out the overlap for anchor, GT box pairs which don't match.
        overlaps *= anchors_matched_gt_box
        # In cases where one anchor matches to 2 GT boxes, we pick the GT box
        # with the highest overlap as a max.
        gt_box_matches_per_anchor = ops.argmax(overlaps, axis=1)
        gt_box_matches_per_anchor_mask = ops.max(overlaps, axis=1) > 0
        # TODO(ianstenbit): Once ops.take_along_axis supports -1 in Torch,
        # replace gt_box_matches_per_anchor with
        # ops.where(
        #     ops.max(overlaps, axis=1) > 0, ops.argmax(overlaps, axis=1), -1
        # )
        # and get rid of the manual masking
        gt_box_matches_per_anchor = ops.cast(gt_box_matches_per_anchor, "int32")

        # We select the GT boxes and labels that correspond to anchor matches.
        bbox_labels = ops.take_along_axis(
            gt_bboxes, gt_box_matches_per_anchor[:, :, None], axis=1
        )
        bbox_labels = ops.where(
            gt_box_matches_per_anchor_mask[:, :, None], bbox_labels, -1
        )
        class_labels = ops.take_along_axis(
            gt_labels, gt_box_matches_per_anchor, axis=1
        )
        class_labels = ops.where(
            gt_box_matches_per_anchor_mask, class_labels, -1
        )

        class_labels = ops.one_hot(
            ops.cast(class_labels, "int32"), self.num_classes
        )

        # Finally, we normalize an anchor's class labels based on the relative
        # strength of the anchors match with the corresponding GT box.
        alignment_metrics *= anchors_matched_gt_box
        max_alignment_per_gt_box = ops.max(
            alignment_metrics, axis=-1, keepdims=True
        )
        max_overlap_per_gt_box = ops.max(overlaps, axis=-1, keepdims=True)

        normalized_alignment_metrics = ops.max(
            alignment_metrics
            * max_overlap_per_gt_box
            / (max_alignment_per_gt_box + self.epsilon),
            axis=-2,
        )
        class_labels *= normalized_alignment_metrics[:, :, None]

        # On TF backend, the final "4" becomes a dynamic shape so we include
        # this to force it to a static shape of 4. This does not actually
        # reshape the Tensor.
        bbox_labels = ops.reshape(bbox_labels, (-1, num_anchors, 4))
        return (
            ops.stop_gradient(bbox_labels),
            ops.stop_gradient(class_labels),
            ops.stop_gradient(
                ops.cast(gt_box_matches_per_anchor > -1, "float32")
            ),
        )

    def call(
        self, scores, decode_bboxes, anchors, gt_labels, gt_bboxes, gt_mask
    ):
        """Computes target boxes and classes for anchors.

        Args:
            scores: a Float Tensor of shape (batch_size, num_anchors,
                num_classes) representing predicted class scores for each
                anchor.
            decode_bboxes: a Float Tensor of shape (batch_size, num_anchors, 4)
                representing predicted boxes for each anchor.
            anchors: a Float Tensor of shape (batch_size, num_anchors, 2)
                representing the xy coordinates of the center of each anchor.
            gt_labels: a Float Tensor of shape (batch_size, num_gt_boxes)
                representing the classes of ground truth boxes.
            gt_bboxes: a Float Tensor of shape (batch_size, num_gt_boxes, 4)
                representing the ground truth bounding boxes in xyxy format.
            gt_mask: A Boolean Tensor of shape (batch_size, num_gt_boxes)
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

        if isinstance(gt_mask, tf.RaggedTensor):
            gt_mask = gt_mask.to_tensor()

        max_num_boxes = ops.shape(gt_bboxes)[1]

        # If there are no GT boxes in the batch, we short-circuit and return
        # empty targets to avoid NaNs.
        return ops.cond(
            ops.array(max_num_boxes > 0),
            lambda: self.assign(
                scores, decode_bboxes, anchors, gt_labels, gt_bboxes, gt_mask
            ),
            lambda: (
                ops.zeros_like(decode_bboxes),
                ops.zeros_like(scores),
                ops.zeros_like(scores[..., 0]),
            ),
        )

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
