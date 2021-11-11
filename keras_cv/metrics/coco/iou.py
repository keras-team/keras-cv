"""Contains functions to compute ious of bounding boxes."""
import tensorflow as tf

from keras_cv.metrics.coco import bbox


def _area(left, right, top, bottom):
    # TODO(lukewood): revisit with user later, should this throw an Error?
    if left >= right:
        return 0.0
    if top >= bottom:
        return 0.0
    width = right - left
    height = bottom - top
    return width * height


def _compute_single_iou(bb1, bb2):
    """computes the intersection over union between bb1 and bb2.

  bb1 and bb2 are expected to come in the 'corners' format, or:
    [left, right, top, bottom].
  For example, the bounding box with it's left side at 100, right side at 200,
  top at 101, and bottom at 201 would be represented as:
    [100, 200, 101, 201]

  Args:
    bb1: Tensor representing the first bounding box in 'corners' format.
    bb2: Tensor representing the second bounding box in 'corners' format.

  Returns:
    iou: Tensor representing the intersection over union of the two bounding
      boxes.
  """
    # bounding box indices

    intersection_left = tf.math.maximum(bb1[bbox.LEFT], bb2[bbox.LEFT])
    intersection_right = tf.math.minimum(bb1[bbox.RIGHT], bb2[bbox.RIGHT])

    intersection_top = tf.math.maximum(bb1[bbox.TOP], bb2[bbox.TOP])
    intersection_bottom = tf.math.minimum(bb1[bbox.BOTTOM], bb2[bbox.BOTTOM])

    area_intersection = _area(
        intersection_left, intersection_right, intersection_top, intersection_bottom
    )
    area_bb1 = _area(
        bb1[bbox.LEFT], bb1[bbox.RIGHT], bb1[bbox.TOP], bb1[bbox.BOTTOM]
    )
    area_bb2 = _area(
        bb2[bbox.LEFT], bb2[bbox.RIGHT], bb2[bbox.TOP], bb2[bbox.BOTTOM]
    )

    area_union = area_bb1 + area_bb2 - area_intersection
    return tf.math.divide_no_nan(area_intersection, area_union)


def compute_ious_for_image(ground_truths, predictions):
    """computes a lookup table vector containing the ious for a given set of ground truth and predicted bounding boxes.

  The lookup vector is to be indexed by [`prediction_number`,
  `ground_truth_number`].  This ordering is chosen to align with that of
  pycocotools, which chose this format.

  Bounding boxes are expected to be in the corners format of [left, bbox.RIGHT,
  bbox.TOP,
  bbox.BOTTOM].  For example, the bounding box with it's left side at 100,
  bbox.RIGHT
  side at 200, bbox.TOP at 101, and bbox.BOTTOM at 201 would be represented as:
  > [100, 200, 101, 201]

  Args:
    ground_truths: a list of bounding boxes in 'corners' format.
    predictions: a list of bounding boxes in 'corners' format.

  Returns:
    iou_lookup_table: a vector containing the pairwise ious of ground_truths and
      predictions.
  """
    ground_truths_size = tf.shape(ground_truths)[0]
    predictions_size = tf.shape(predictions)[0]
    result = tf.TensorArray(
        tf.float32, size=predictions_size * ground_truths_size, dynamic_size=False
    )
    # populate [m, n] ious
    for g in tf.range(ground_truths_size):
        # TODO(lukewood): revisit performance here
        for p in tf.range(predictions_size):
            target_index = g * predictions_size + p
            result = result.write(
                target_index, _compute_single_iou(predictions[p], ground_truths[g])
            )
    return tf.reshape(result.stack(), (ground_truths_size, predictions_size))
