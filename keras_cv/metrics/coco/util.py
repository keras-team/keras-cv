"""Contains shared utilities for Keras COCO metrics."""
import tensorflow as tf

from keras_cv.metrics.coco import bbox


def filter_boxes(boxes, value, axis=4):
    """filter_boxes is used to sort a list of bounding boxes by a given axis.
    The most common use case for this is to filter out to get a specific bbox.CLASS.
    Args:
        boxes: Tensor of bounding boxes in format `[images, bboxes, 6]`
        value: Value the specified axis must match
        axis: Integer identifying the axis on which to sort, default 4
    Returns:
        boxes: A new Tensor of bounding boxes, where boxes[axis]==value
    """
    return tf.gather_nd(tf.where(boxes[axis] == value))


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
