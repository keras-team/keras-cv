import tensorflow as tf

from keras_cv.metrics.coco.base import COCOBase


class COCORecall(COCOBase):
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
