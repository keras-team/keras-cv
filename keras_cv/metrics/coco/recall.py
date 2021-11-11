"""Contains the COCORecall class.

This class takes average recall across categories, iou thresholds and images.
"""
import tensorflow as tf
import tensorflow.keras.initializers as initializers

from keras_cv.metrics.coco import bbox
from keras_cv.metrics.coco import iou as iou_lib
from keras_cv.metrics.coco import util


class COCORecall(tf.keras.metrics.Metric):
    """Computes the COCO Average Recall across IoU thresholds and categories.

    Args:
      iou_thresholds: iterable of values for use as IoU thresholds.  The default
        value is the range [0.5:0.95,0.05].
      categories: list of categories, or number of categories to use.
  """

    def __init__(self, iou_thresholds=None, categories=None, **kwargs):
        super(COCORecall, self).__init__(**kwargs)
        iou_thresholds = iou_thresholds or [x / 100.0 for x in range(50, 100, 5)]
        self.iou_thresholds = self.add_weight(
            name="iou_thresholds",
            shape=(len(iou_thresholds),),
            initializer=initializers.Constant(iou_thresholds),
            dtype=tf.float32,
        )
        self.categories = self.add_weight(
            name="categories",
            shape=(len(categories),),
            initializer=initializers.Constant(categories),
            dtype=tf.float32,
        )
        self.count = self.add_weight(
            name="count", shape=(), initializer="zeros", dtype=tf.float32
        )
        self.recall_sum = self.add_weight(
            name="recall_sum", shape=(), initializer="zeros", dtype=tf.float32
        )

    # TODO(lukewood): add tf.function(jit_compile=True) to this
    def update_state(self, y_true, y_pred, sample_weight=None):
        # sort predictions based on confidence
        num_images = y_true.shape[0]
        y_pred = util.sort_bboxes(y_pred, axis=bbox.CONFIDENCE)

        # compute ious per image
        # TODO(lukewood): replace dict with a Tensor, images are sequential anyways
        ious = tf.TensorArray(tf.float32, size=num_images, dynamic_size=False)
        for i in tf.range(num_images):
            ious = ious.write(
                i, iou_lib.compute_ious_for_image(y_true[i], y_pred[i])
            )
        # iou lookup [image, bbox_true, bbox_pred]
        ious = ious.stack()

        # for each category, for each image, compute the recall score
        num_thresholds = len(self.iou_thresholds)
        num_categories = len(self.categories)

        # TODO(lukewood): support area ranges (min_size, max_size)
        # TODO(lukewood): support min/max detections

        # Iteration order is very important, we track the number of examples
        # internally in order to give all samples taken equal weight, due to this we
        # must iterate over images first, then thresholds/categories internally.  We
        # find the means by using the tf.reduce_mean below.
        recall_result = tf.TensorArray(
            tf.float32, size=num_images, dynamic_size=False
        )
        for image in tf.range(num_images):
            img_result = tf.TensorArray(
                tf.float32, size=num_thresholds, dynamic_size=False
            )
            for iou_idx in tf.range(num_thresholds):
                iou_thr = self.iou_thresholds[iou_idx]
                iou_result = tf.TensorArray(
                    tf.float32, size=num_categories, dynamic_size=False
                )
                for category_idx in tf.range(num_categories):
                    category = self.categories[category_idx]
                    result = self._single_image_recall(
                        y_true[image],
                        y_pred[image],
                        ious[image, :, :],
                        iou_thr,
                        category,
                        image,
                    )
                    iou_result = iou_result.write(category_idx, result)
                img_result = img_result.write(iou_idx, iou_result.stack())
            recall_result = recall_result.write(image, img_result.stack())
        recall_result = recall_result.stack()

        self.count.assign_add(num_images)
        # average over categories, average over thresholds, then sum over images
        # TODO(lukewood): boolean mask out -1 values

        num_present_categories = _count_not_matching(recall_result[:, 0, :], -1)

        # ReLu is used here to filter out the sentinel -1s in a performant way.
        recall_summed = tf.reduce_sum(tf.nn.relu(recall_result), [-1, -2])
        recall_mean = recall_summed / tf.cast(
            num_thresholds * num_present_categories, dtype=recall_summed.dtype
        )
        self.recall_sum.assign_add(tf.reduce_sum(recall_mean))

    def _single_image_recall(
        self, y_true, y_pred, iou_table, iou_thr, category, image
    ):
        # y_true: [bboxes, 5]
        # y_pred: [bboxes, 6]

        found_bboxes = 0.0

        num_true_bbox = tf.shape(y_true)[0]
        num_pred_bbox = tf.shape(y_pred)[0]

        num_true_bbox_matching = _count_matching(y_true[:, bbox.CLASS], category)

        # TODO(lukewood): optimize by using tf.gather, tf.where, etc to batch these
        # loops
        # TODO(lukewood): additionally we will need to perform masking over the iou
        # table by category, maybe it's actually better to create the iou table by
        # category in the first place.
        if num_true_bbox_matching == 0:
            # undefined, -1 is a sentinel
            return -1.0

        for true_idx in tf.range(num_true_bbox):
            if y_true[true_idx][bbox.CLASS] != category:
                continue
            for pred_idx in tf.range(num_pred_bbox):
                if y_pred[pred_idx][bbox.CLASS] != category:
                    continue
                if iou_table[true_idx, pred_idx] >= iou_thr:
                    found_bboxes += 1.0
                    break

        return found_bboxes / tf.cast(num_true_bbox_matching, dtype=tf.float32)

    def reset_state(self):
        self.count.assign(0.0)
        self.recall_sum.assign(0.0)

    def result(self):
        return self.recall_sum / self.count


def _count_not_matching(x, value):
    return tf.math.count_nonzero(tf.where(tf.equal(x, value), 0.0, 1.0), axis=-1)


def _count_matching(x, value):
    return tf.math.count_nonzero(tf.where(tf.equal(x, value), 1.0, 0.0), axis=-1)
