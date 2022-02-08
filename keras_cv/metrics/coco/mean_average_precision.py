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
from numpy import pad
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.initializers as initializers

from keras_cv.metrics.coco import iou as iou_lib
from keras_cv.metrics.coco import utils
from keras_cv.utils import bbox


class COCOMeanAveragePrecision(tf.keras.metrics.Metric):
    """COCOMeanAveragePrecision computes MaP.

    Args:
        iou_thresholds: defaults to [0.5:0.05:0.95].
        category_ids: no default, users must provide.
        area_range: area range to consider bounding boxes in. Defaults to all.
        max_detections: number of maximum detections a model is allowed to make.
        recall_thresholds: List of floats.  Defaults to [0:.01:1].
    """

    def __init__(
        self,
        category_id=1,
        recall_thresholds=None,
        iou_threshold=0.5,
        area_range=(0, 1e9**2),
        max_detections=100,
        num_buckets=10000,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Initialize parameter values
        self.iou_threshold = iou_threshold
        self.area_range = area_range
        self.max_detections = max_detections
        self.category_id = category_id
        self.recall_thresholds = recall_thresholds or [x / 100 for x in range(0, 101)]
        self.num_buckets = num_buckets

        self.ground_truths = self.add_weight(
            "ground_truths", dtype=tf.int32, initializer="zeros"
        )
        self.true_positive_buckets = self.add_weight(
            "true_positive_buckets",
            shape=(num_buckets,),
            dtype=tf.int32,
            initializer="zeros",
        )
        self.false_positive_buckets = self.add_weight(
            "true_positive_buckets",
            shape=(num_buckets,),
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

        for img in tf.range(num_images):
            ground_truths = utils.filter_out_sentinels(y_true[img])
            detections = utils.filter_out_sentinels(y_pred[img])
            ground_truths = utils.filter_boxes_by_area_range(
                ground_truths, self.area_range[0], self.area_range[1]
            )
            detections = utils.filter_boxes_by_area_range(
                detections, self.area_range[0], self.area_range[1]
            )
            ground_truths = utils.filter_boxes(
                ground_truths, value=self.category_id, axis=bbox.CLASS
            )
            detections = utils.filter_boxes(
                detections, value=self.category_id, axis=bbox.CLASS
            )

            ious = iou_lib.compute_ious_for_image(ground_truths, detections)

            pred_matches = utils.match_boxes(
                ground_truths, detections, self.iou_threshold, ious
            )
            dt_scores = detections[:, bbox.CONFIDENCE]
            indices = tf.argsort(dt_scores, direction="DESCENDING")

            dt_scores = tf.gather(dt_scores, indices)
            dtm = tf.gather(pred_matches, indices)

            tps = tf.cast(dtm != -1, tf.int32)
            fps = tf.cast(dtm == -1, tf.int32)

            confidence_buckets = tf.math.floor(self.num_buckets * dt_scores, tf.int32)

            # TODO(lukewood): vectorize this operation.  We should be able to
            # using a mask constructed as follows:
            # bucket_mask = tf.one_hot(tf.range(0, self.num_buckets), self.num_buckets)
            # alongside with TensorFlow's broadcasting rules.

            tps_by_bucket = tf.TensorArray(tf.int32, size=self.num_buckets)
            fps_by_bucket = tf.TensorArray(tf.int32, size=self.num_buckets)
            for bucket in range(self.num_buckets):
                indices = tf.where(confidence_buckets == bucket)
                tps_for_bucket = tf.gather(tps, indices)
                fps_for_bucket = tf.gather(fps, indices)
                tps_by_bucket = tps_by_bucket.write(
                    bucket, tf.math.reduce_sum(tps_for_bucket)
                )
                fps_for_bucket = fps_by_bucket.write(
                    bucket, tf.math.reduce_sum(fps_for_bucket)
                )

            self.true_positive_buckets.assign_add(tps_by_bucket.stack())
            self.false_positive_buckets.assign_add(fps_by_bucket.stack())

            self.ground_truths.assign_add(tf.cast(tf.shape(ground_truths)[0], tf.int32))

    def result(self):
        tp_sum = tf.cumsum(self.true_positive_buckets, axis=-1)
        fp_sum = tf.cumsum(self.false_positive_buckets, axis=-1)

        rc = tp_sum / self.ground_truths
        pr = tp_sum / (fp_sum + tp_sum)
        inds = tf.searchsorted(rc, tf.constant(self.recall_thresholds), side="left")
        precision_result = tf.TensorArray(tf.float32, size=len(self.recall_thresholds))
        for ri in tf.range(len(self.recall_thresholds)):
            pi = inds[ri]
            if pi < tf.shape(pr)[0]:
                pr_res = pr[pi]
                precision_result = precision_result.write(ri, pr_res)

        pr_per_threshold = precision_result.stack()
        return tf.math.reduce_mean(pr_per_threshold)
