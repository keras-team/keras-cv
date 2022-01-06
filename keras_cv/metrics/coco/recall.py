import tensorflow as tf

from keras_cv.metrics.coco.base import COCOBase


class COCORecall(COCOBase):
    def result(self):
        shape = tf.shape(self.true_positives)
        a = shape[2]
        m = shape[3]
        n_results = a * m

        if n_results == 1:
            return self._single_result(0, 0)

        result = {}
        for a_i in tf.range(a):
            for m_i in tf.range(m):
                key = self._key_for(a_i, m_i)
                result[key] = self._single_result(a_i, m_i)
        return result

    def _single_result(self, a_i, m_i):
        # TODO(lukewood): do I need to mask out -1s???
        # TODO(lukewood): do I need to mask out NaNs?
        present_values = self.ground_truth_boxes[:, a_i] != 0
        n_present_categories = tf.math.reduce_sum(tf.cast(present_values, tf.float32), axis=-1)
        if n_present_categories == 0.:
            return 0.0

        recalls = tf.math.divide_no_nan(self.true_positives[:, :, a_i, m_i], self.ground_truth_boxes[None, :, a_i])
        recalls_per_threshold = tf.math.reduce_sum(recalls, axis=-1) / n_present_categories
        return tf.math.reduce_mean(recalls_per_threshold)

    def _key_for(self, a_i, m_i):
        # TODO(lukewood): format like the real coco metrics
        area_range = self.area_ranges[a_i]
        max_dets = self.max_detections[m_i]
        return f"Recall @ {self.iou_threshold_str_rep()}, {area_range}, max_dets={max_dets}"

    def iou_threshold_str_rep(self):
        # TODO(lukewood): generate a nice value
        if len(self._user_iou_thresholds) == 1:
            return f"[{self._user_iou_thresholds[0]}"
        return f"[{self._user_iou_thresholds[0]}, {self._user_iou_thresholds[-1]}, {self._user_iou_thresholds[1]-self._user_iou_thresholds[0]}]"
