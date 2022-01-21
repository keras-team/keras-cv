"""Contains functions to compute ious of bounding boxes."""
import tensorflow as tf

from keras_cv.util import bbox


def _compute_single_iou(bboxes1, bb2):
    """computes the intersection over union between bboxes1 and bb2.

    bboxes1 and bb2 are expected to come in the 'corners' format, or:
      [left, right, top, bottom].
    For example, the bounding box with it's left side at 100, right side at 200,
    top at 101, and bottom at 201 would be represented as:
      [100, 200, 101, 201]

    Args:
      bboxes1: Tensor representing the first bounding box in 'corners' format.
      bb2: Tensor representing the second bounding box in 'corners' format.

    Returns:
      iou: Tensor representing the intersection over union of the two bounding
        boxes.
    """
    # bounding box indices

    intersection_left = tf.math.maximum(bboxes1[bbox.LEFT], bb2[bbox.LEFT])
    intersection_right = tf.math.minimum(bboxes1[bbox.RIGHT], bb2[bbox.RIGHT])

    intersection_top = tf.math.maximum(bboxes1[bbox.TOP], bb2[bbox.TOP])
    intersection_bottom = tf.math.minimum(bboxes1[bbox.BOTTOM], bb2[bbox.BOTTOM])

    area_intersection = _area(
        intersection_left, intersection_right, intersection_top, intersection_bottom
    )
    area_bboxes1 = _area(
        bb1[bbox.LEFT], bb1[bbox.RIGHT], bb1[bbox.TOP], bb1[bbox.BOTTOM]
    )
    area_bb2 = _area(
        bb2[bbox.LEFT], bb2[bbox.RIGHT], bb2[bbox.TOP], bb2[bbox.BOTTOM]
    )

    area_union = area_bboxes1 + area_bb2 - area_intersection
    return tf.math.divide_no_nan(area_intersection, area_union)


def compute_ious_for_image(boxes1, boxes2):
    """computes a lookup table vector containing the ious for a given set boxes.

    The lookup vector is to be indexed by [`boxes1_index`,`boxes2_index`].

    Bounding boxes are expected to be in the corners format of `[bbox.LEFT, bbox.RIGHT,
    bbox.TOP, bbox.BOTTOM]`.  For example, the bounding box with it's left side at 100,
    bbox.RIGHT side at 200, bbox.TOP at 101, and bbox.BOTTOM at 201 would be represented
    as:
    > [100, 200, 101, 201]

    Args:
      boxes1: a list of bounding boxes in 'corners' format.
      boxes2: a list of bounding boxes in 'corners' format.

    Returns:
      iou_lookup_table: a vector containing the pairwise ious of boxes1 and
        boxes2.
    """
    zero = tf.convert_to_tensor(0.0, boxes1.dtype)
    boxes1_ymin, boxes1_xmin, boxes1_ymax, boxes1_xmax = tf.unstack(
        boxes1[..., :4, None], 4, axis=-2
    )
    boxes2_ymin, boxes2_xmin, boxes2_ymax, boxes2_xmax = tf.unstack(
        boxes2[None, ..., :4], 4, axis=-1
    )
    boxes1_width = tf.maximum(zero, boxes1_xmax - boxes1_xmin)
    boxes1_height = tf.maximum(zero, boxes1_ymax - boxes1_ymin)
    boxes2_width = tf.maximum(zero, boxes2_xmax - boxes2_xmin)
    boxes2_height = tf.maximum(zero, boxes2_ymax - boxes2_ymin)
    boxes1_area = boxes1_width * boxes1_height
    boxes2_area = boxes2_width * boxes2_height
    intersect_ymin = tf.maximum(boxes1_ymin, boxes2_ymin)
    intersect_xmin = tf.maximum(boxes1_xmin, boxes2_xmin)
    intersect_ymax = tf.minimum(boxes1_ymax, boxes2_ymax)
    intersect_xmax = tf.minimum(boxes1_xmax, boxes2_xmax)
    intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
    intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height

    union_area = boxes1_area + boxes2_area - intersect_area
    return tf.math.divide_no_nan(intersect_area, union_area)
