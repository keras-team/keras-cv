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
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.initializers as initializers
from numpy import pad

from keras_cv.utils import iou as iou_lib
from keras_cv.metrics.coco import utils
from keras_cv.utils import bbox


class COCOMeanAveragePrecision(tf.keras.metrics.Metric):
    """COCOMeanAveragePrecision computes MaP.

    Args:
        iou_thresholds: defaults to [0.5:0.05:0.95].
        class_ids: no default, users must provide.
        area_range: area range to consider bounding boxes in. Defaults to all.
        max_detections: number of maximum detections a model is allowed to make.
        recall_thresholds: List of floats.  Defaults to [0:.01:1].
    """

    def __init__(
        self,
        category_ids,
        recall_thresholds=None,
        iou_thresholds=None,
        area_range=None,
        max_detections=100,
        num_buckets=10000,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Initialize parameter values
        self.iou_thresholds = iou_thresholds or [x / 100.0 for x in range(50, 100, 5)]
        self.area_range = area_range
        self.max_detections = max_detections
        self.category_ids = category_ids
        self.recall_thresholds = recall_thresholds or [x / 100 for x in range(0, 101)]
        self.num_buckets = num_buckets

        self.num_iou_thresholds = len(self.iou_thresholds)
        self.num_category_ids = len(self.category_ids)

        self.ground_truths = self.add_weight("ground_truths", shape=(self.num_category_ids,), dtype=tf.int32, initializer="zeros")
        self.true_positive_buckets = self.add_weight(
            "true_positive_buckets",
            shape=(self.num_category_ids, self.num_iou_thresholds, num_buckets,),
            dtype=tf.int32,
            initializer="zeros",
        )
        self.false_positive_buckets = self.add_weight(
            "true_positive_buckets",
            shape=(self.num_category_ids, self.num_iou_thresholds, num_buckets,),
            dtype=tf.int32,
            initializer="zeros",
        )

    def reset_state(self):
        self.true_positive_buckets.assign(tf.zeros_like(self.true_positive_buckets))
        self.false_positive_buckets.assign(tf.zeros_like(self.false_positive_buckets))
        self.ground_truths.assign(tf.zeros_like(self.ground_truths))

    def update_state(self, y_true, y_pred):
        num_images = tf.shape(y_true)[0]

        y_pred = utils.sort_bboxes(y_pred, axis=bbox.CONFIDENCE)

        ground_truth_boxes_update = tf.zeros_like(self.ground_truths)
        true_positive_buckets_update = tf.zeros_like(self.true_positive_buckets)
        false_positive_buckets_update = tf.zeros_like(self.false_positive_buckets)

        for img in tf.range(num_images):
            ground_truths = utils.filter_out_sentinels(y_true[img])
            detections = utils.filter_out_sentinels(y_pred[img])

            if self.area_range is not None:
                ground_truths = utils.filter_boxes_by_area_range(
                    ground_truths, self.area_range[0], self.area_range[1]
                )
                detections = utils.filter_boxes_by_area_range(
                    detections, self.area_range[0], self.area_range[1]
                )

            true_positives_update = tf.TensorArray(tf.int32, size=self.num_category_ids * self.num_iou_thresholds)
            false_positives_update = tf.TensorArray(tf.int32, size=self.num_category_ids * self.num_iou_thresholds)
            ground_truths_update = tf.TensorArray(tf.int32, size=self.num_category_ids)

            for c_i in range(self.num_category_ids):
                category_id = self.category_ids[c_i]
                ground_truths = utils.filter_boxes(
                    ground_truths, value=category_id, axis=bbox.CLASS
                )
                detections = utils.filter_boxes(
                    detections, value=category_id, axis=bbox.CLASS
                )
                ground_truths_update = ground_truths_update.write(c_i, tf.shape(ground_truths)[0])

                ious = iou_lib.compute_ious_for_image(ground_truths, detections)

                for iou_i in range(self.num_iou_thresholds):
                    iou_threshold = self.iou_thresholds[iou_i]

                    pred_matches = utils.match_boxes(
                        ious, iou_threshold
                    )
                    dt_scores = detections[:, bbox.CONFIDENCE]
                    indices = tf.argsort(dt_scores, direction="DESCENDING")

                    dt_scores = tf.gather(dt_scores, indices)
                    dtm = tf.gather(pred_matches, indices)

                    tps = dtm != -1
                    fps = dtm == -1

                    confidence_buckets = tf.cast(tf.math.floor(self.num_buckets * dt_scores), tf.int32)

                    tps_by_bucket = tf.gather_nd(confidence_buckets, indices=tf.where(tps))
                    fps_by_bucket = tf.gather_nd(confidence_buckets, indices=tf.where(fps))

                    tp_counts_per_bucket = tf.math.bincount(
                        tps_by_bucket, minlength=self.num_buckets, maxlength=self.num_buckets
                    )
                    fp_counts_per_bucket = tf.math.bincount(
                        fps_by_bucket, minlength=self.num_buckets, maxlength=self.num_buckets
                    )
                    true_positives_update = true_positives_update.write((self.num_iou_thresholds * c_i) + iou_i, tp_counts_per_bucket)
                    false_positives_update = false_positives_update.write((self.num_iou_thresholds * c_i) + iou_i, fp_counts_per_bucket)

            true_positives_update = tf.reshape(true_positives_update.stack(), (self.num_category_ids, self.num_iou_thresholds, self.num_buckets))
            false_positives_update = tf.reshape(false_positives_update.stack(), (self.num_category_ids, self.num_iou_thresholds, self.num_buckets))

            true_positive_buckets_update = true_positive_buckets_update + true_positives_update
            false_positive_buckets_update = false_positive_buckets_update + false_positives_update
            ground_truth_boxes_update = ground_truth_boxes_update + ground_truths_update.stack()

        self.ground_truths.assign_add(ground_truth_boxes_update)
        self.true_positive_buckets.assign_add(true_positive_buckets_update)
        self.false_positive_buckets.assign_add(false_positive_buckets_update)

    def result(self):
        true_positives = tf.cast(self.true_positive_buckets, self.dtype)
        false_positivves = tf.cast(self.false_positive_buckets, self.dtype)
        ground_truths = tf.cast(self.ground_truths, self.dtype)

        tp_sum = tf.cumsum(true_positives, axis=-1)
        fp_sum = tf.cumsum(false_positivves, axis=-1)

        present_categories = tf.math.reduce_sum(tf.cast(ground_truths != 0, tf.int32))

        if present_categories == 0:
            return 0.0

        # tp_sum shape, [categories, iou_thr, n_buckets]
        rc = tf.math.divide_no_nan(tp_sum, ground_truths[:, None, None])
        pr = tf.math.divide_no_nan(tp_sum, (fp_sum + tp_sum))

        result = tf.TensorArray(tf.float32, size=self.num_category_ids*self.num_iou_thresholds)
        for i in range(self.num_category_ids):
            for j in range(self.num_iou_thresholds):
                rc_i = rc[i, j]
                pr_i = pr[i, j]
                inds = tf.searchsorted(rc_i, tf.constant(self.recall_thresholds), side="left")

                precision_result = tf.TensorArray(self.dtype, size=len(self.recall_thresholds))
                for ri in tf.range(len(self.recall_thresholds)):
                    pi = inds[ri]
                    if pi < tf.shape(pr_i)[0]:
                        pr_res = pr_i[pi]
                        precision_result = precision_result.write(ri, pr_res)

                pr_per_recall_threshold = precision_result.stack()
                result_ij = tf.math.reduce_mean(pr_per_recall_threshold, axis=-1)
                result = result.write(j + i*self.num_iou_thresholds, result_ij)

        result = tf.reshape(result.stack(), (self.num_category_ids, self.num_iou_thresholds))
        result = tf.math.reduce_mean(result, axis=-1)
        return tf.math.reduce_sum(result, axis=0) / tf.cast(present_categories, tf.float32)
