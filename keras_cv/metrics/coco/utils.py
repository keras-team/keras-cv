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

from keras_cv import bounding_box


def filter_boxes_by_area_range(boxes, min_area, max_area):
    areas = bounding_box_area(boxes)
    inds = tf.where(tf.math.logical_and(areas >= min_area, areas < max_area))
    return tf.gather_nd(boxes, inds)


def bounding_box_area(boxes):
    """box_areas returns the area of the provided bounding boxes.
    Args:
        boxes: Tensor of bounding boxes of shape `[..., 4+]` in corners format.
    Returns:
        areas: Tensor of areas of shape `[...]`.
    """
    w = boxes[..., bounding_box.XYXY.RIGHT] - boxes[..., bounding_box.XYXY.LEFT]
    h = boxes[..., bounding_box.XYXY.BOTTOM] - boxes[..., bounding_box.XYXY.TOP]
    return tf.math.multiply(w, h)


def filter_boxes(boxes, value, axis=4):
    """filter_boxes is used to select only boxes matching a given class.
    The most common use case for this is to filter to accept only a specific
    bounding_box.CLASS.
    Args:
        boxes: Tensor of bounding boxes in format `[images, bounding_boxes, 6]`
        value: Value the specified axis must match
        axis: Integer identifying the axis on which to sort, default 4
    Returns:
        boxes: A new Tensor of bounding boxes, where boxes[axis]==value
    """
    return tf.gather_nd(boxes, tf.where(boxes[:, axis] == value))


def to_sentinel_padded_bounding_box_tensor(box_sets):
    """pad_with_sentinels returns a Tensor of bounding_boxes padded with -1s
    to ensure that each bounding_box set has identical dimensions.  This is to
    be used before passing bounding_box predictions, or bounding_box ground truths to
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
    or ground truth bounding_box tensor to ensure dimensions match.
    Args:
        boxes: Tensor of bounding boxes in format `[bounding_boxes, 6]`, usually from a
            single image.
    Returns:
        boxes: A new Tensor of bounding boxes, where boxes[axis]!=-1.
    """
    return tf.gather_nd(boxes, tf.where(boxes[:, bounding_box.XYXY.CLASS] != -1))


def sort_bounding_boxes(boxes, axis=5):
    """sort_bounding_boxes is used to sort a list of bounding boxes by a given axis.

    The most common use case for this is to sort by bounding_box.XYXY.CONFIDENCE, as
    this is a part of computing both COCORecall and COCOMeanAveragePrecision.
    Args:
        boxes: Tensor of bounding boxes in format `[images, bounding_boxes, 6]`
        axis: Integer identifying the axis on which to sort, default 5
    Returns:
        boxes: A new Tensor of Bounding boxes, sorted on an image-wise basis.
    """
    num_images = tf.shape(boxes)[0]
    boxes_sorted_list = tf.TensorArray(tf.float32, size=num_images, dynamic_size=False)
    for img in tf.range(num_images):
        preds_for_img = boxes[img, :, :]
        prediction_scores = preds_for_img[:, axis]
        _, idx = tf.math.top_k(prediction_scores, tf.shape(preds_for_img)[0])
        boxes_sorted_list = boxes_sorted_list.write(
            img, tf.gather(preds_for_img, idx, axis=0)
        )

    return boxes_sorted_list.stack()


def match_boxes(ious, threshold):
    """matches bounding boxes from y_true to boxes in y_pred.

    Args:
        ious: lookup table from [y_true, y_pred] => IoU.
        threshold: minimum IoU for a pair to be considered a match.
    Returns:
        a mapping from [y_pred] => matching y_true index.  Dimension
        of result tensor is equal to the number of boxes in y_pred.
    """
    num_true = tf.shape(ious)[0]
    num_pred = tf.shape(ious)[1]

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

            if ious[gt_idx, detection_idx] < iou:
                continue

            iou = ious[gt_idx, detection_idx]
            match_index = gt_idx

        # Write back the match indices
        pred_matches = pred_matches.write(detection_idx, match_index)
        if match_index == -1:
            continue
        gt_matches = gt_matches.write(match_index, detection_idx)
    return pred_matches.stack()
