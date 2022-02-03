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
"""Contains shared utilities for Keras COCO metrics."""
import tensorflow as tf

from keras_cv.utils import bbox


def filter_boxes_by_area_range(boxes, min_area, max_area):
    areas = bbox_area(boxes)
    inds = tf.where(tf.math.logical_and(areas >= min_area, areas < max_area))
    return tf.gather_nd(boxes, inds)


def bbox_area(boxes):
    """box_areas returns the area of the provided bounding boxes.
    Args:
        boxes: Tensor of bounding boxes of shape `[..., 4+]` in corners format.
    Returns:
        areas: Tensor of areas of shape `[...]`.
    """
    w = boxes[..., bbox.RIGHT] - boxes[..., bbox.LEFT]
    h = boxes[..., bbox.BOTTOM] - boxes[..., bbox.TOP]
    return tf.math.multiply(w, h)


def filter_boxes(boxes, value, axis=4):
    """filter_boxes is used to select only boxes matching a given class.
    The most common use case for this is to filter to accept only a specific
    bbox.CLASS.
    Args:
        boxes: Tensor of bounding boxes in format `[images, bboxes, 6]`
        value: Value the specified axis must match
        axis: Integer identifying the axis on which to sort, default 4
    Returns:
        boxes: A new Tensor of bounding boxes, where boxes[axis]==value
    """
    return tf.gather_nd(boxes, tf.where(boxes[:, axis] == value))


def to_sentinel_padded_bbox_tensor(box_sets):
    """pad_with_sentinels returns a Tensor of bboxes padded with -1s
    to ensure that each bbox set has identical dimensions.  This is to
    be used before passing bbox predictions, or bbox ground truths to
    the keras COCO metrics.
    Args:
        box_sets: List of Tensors representing bounding boxes, or a list of lists of
            Tensors.
    Returns:
        boxes: A new Tensor where each value missing is populated with -1.
    """
    return tf.ragged.stack(box_sets).to_tensor(default_value=-1)


def filter_out_sentinels(boxes):
    """filter_out_sentinels to filter out boxes that were padded on to the prediction
    or ground truth bbox tensor to ensure dimensions match.
    Args:
        boxes: Tensor of bounding boxes in format `[bboxes, 6]`, usually from a
            single image.
    Returns:
        boxes: A new Tensor of bounding boxes, where boxes[axis]!=-1.
    """
    return tf.gather_nd(boxes, tf.where(boxes[:, bbox.CLASS] != -1))


def sort_bboxes(boxes, axis=5):
    """sort_bboxes is used to sort a list of bounding boxes by a given axis.
    The most common use case for this is to sort by bbox.CONFIDENCE, as this is a
    part of computing both COCORecall and COCOMeanAveragePrecision.
    Args:
        boxes: Tensor of bounding boxes in format `[images, bboxes, 6]`
        axis: Integer identifying the axis on which to sort, default 5
    Returns:
        boxes: A new Tensor of Bounding boxes, sorted on an image-wise basis.
    """
    num_images = tf.shape(boxes)[0]
    boxes_sorted_list = tf.TensorArray(tf.float32, size=num_images, dynamic_size=False)
    for img in tf.range(num_images):
        preds_for_img = boxes[img, :, :]
        prediction_scores = preds_for_img[:, axis]
        _, idx = tf.math.top_k(prediction_scores, preds_for_img.shape[0])
        boxes_sorted_list = boxes_sorted_list.write(
            img, tf.gather(preds_for_img, idx, axis=0)
        )

    return boxes_sorted_list.stack()


def match_boxes(y_true, y_pred, threshold, ious):
    """matches bounding boxes from y_true to boxes in y_pred.

    Args:
        y_true: bounding box tensor of shape [num_boxes, 4+].
        y_pred: bounding box tensor of shape [num_boxes, 4+].
        threshold: minimum IoU for a pair to be considered a match.
        ious: lookup table from [y_true, y_pred] => IoU.
    Returns:
        indices of matches between y_pred and y_true.
    """
    num_true = tf.shape(y_true)[0]
    num_pred = tf.shape(y_pred)[0]

    gt_matches = tf.TensorArray(
        tf.int32,
        size=num_true,
        dynamic_size=False,
        infer_shape=False,
        element_shape=(),
    )
    pred_matches = tf.TensorArray(
        tf.int32,
        size=num_pred,
        dynamic_size=False,
        infer_shape=False,
        element_shape=(),
    )
    for i in tf.range(num_true):
        gt_matches = gt_matches.write(i, -1)
    for i in tf.range(num_pred):
        pred_matches = pred_matches.write(i, -1)

    for detection_idx in tf.range(num_pred):
        match_index = -1
        iou = tf.math.minimum(threshold, 1 - 1e-10)

        for gt_idx in tf.range(num_true):
            if gt_matches.gather([gt_idx]) > -1:
                continue
            # TODO(lukewood): update clause to account for gtIg
            # if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:

            if not ious[gt_idx, detection_idx] >= iou:
                continue

            iou = ious[gt_idx, detection_idx]
            match_index = gt_idx

        # Write back the match indices
        pred_matches = pred_matches.write(detection_idx, match_index)
        if match_index == -1:
            continue
        gt_matches = gt_matches.write(match_index, detection_idx)
    return pred_matches.stack()

def result(self):
    present_values = self.ground_truth_boxes != 0
    n_present_categories = tf.math.reduce_sum(
        tf.cast(present_values, tf.float32), axis=-1
    )
    if n_present_categories == 0.0:
        return 0.0

    recalls = tf.math.divide_no_nan(
        self.true_positives, self.ground_truth_boxes[None, :]
    )
    recalls_per_threshold = (
        tf.math.reduce_sum(recalls, axis=-1) / n_present_categories
    )
    return tf.math.reduce_mean(recalls_per_threshold)
