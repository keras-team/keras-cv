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

from keras_cv.metrics.coco import utils
from keras_cv.utils import bbox
from keras_cv.utils import iou as iou_lib


class COCOMeanAveragePrecision(tf.keras.metrics.Metric):
    """COCOMeanAveragePrecision computes MaP.

    Args:
        class_ids: The class IDs to evaluate the metric for.  To evaluate for
            all classes in over a set of sequentially labelled classes, pass
            `range(num_classes)`.
        iou_thresholds: IoU thresholds over which to evaluate the recall.  Must
            be a tuple of floats, defaults to [0.5:0.05:0.95].
        area_range: area range to constrict the considered bounding boxes in
            metric computation. Defaults to `None`, which makes the metric
            count all bounding boxes.  Must be a tuple of floats.  The first
            number in the tuple represents a lower bound for areas, while the
            second value represents an upper bound.  For example, when
            `(0, 32**2)` is passed to the metric, recall is only evaluated for
            objects with areas less than `32*32`.  If `(32**2, 1000000**2)` is
            passed the metric will only be evaluated for boxes with areas larger
            than `32**2`, and smaller than `1000000**2`.
        max_detections: number of maximum detections a model is allowed to make.
            Must be an integer, defaults to `100`.
        recall_thresholds: The list of thresholds to average over in the MaP
            computation.  List of floats.  Defaults to [0:.01:1].

    Usage:

    COCOMeanAveragePrecision accepts two Tensors as input to it's `update_state` method.
    These Tensors represent bounding boxes in `corners` format.  Utilities
    to convert Tensors from `xywh` to `corners` format can be found in
    `keras_cv.utils.bbox`.

    Each image in a dataset may have a different number of bounding boxes,
    both in the ground truth dataset and the prediction set.  In order to
    account for this, you may either pass a `tf.RaggedTensor`, or pad Tensors
    with `-1`s to indicate unused boxes.  A utility function to perform this
    padding is available at `keras_cv_.utils.bbox.pad_bbox_batch_to_shape`.

    ```python
    coco_map = keras_cv.metrics.COCOMeanAveragePrecision(
        max_detections=100,
        class_ids=[1]
    )

    y_true = np.array([[[0, 0, 10, 10, 1], [20, 20, 10, 10, 1]]]).astype(np.float32)
    y_pred = np.array([[[0, 0, 10, 10, 1, 1.0], [5, 5, 10, 10, 1, 0.9]]]).astype(
        np.float32
    )
    coco_map.update_state(y_true, y_pred)
    coco_map.result()
    # TODO(lukewood) print result before submitting
    ```
    """

    def __init__(
        self,
        class_ids,
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
        self.class_ids = class_ids
        self.recall_thresholds = recall_thresholds or [x / 100 for x in range(0, 101)]
        self.num_buckets = num_buckets

        self.num_iou_thresholds = len(self.iou_thresholds)
        self.num_class_ids = len(self.class_ids)

        self.ground_truths = self.add_weight(
            "ground_truths",
            shape=(self.num_class_ids,),
            dtype=tf.int32,
            initializer="zeros",
        )
        self.true_positive_buckets = self.add_weight(
            "true_positive_buckets",
            shape=(
                self.num_class_ids,
                self.num_iou_thresholds,
                num_buckets,
            ),
            dtype=tf.int32,
            initializer="zeros",
        )
        self.false_positive_buckets = self.add_weight(
            "false_positive_buckets",
            shape=(
                self.num_class_ids,
                self.num_iou_thresholds,
                num_buckets,
            ),
            dtype=tf.int32,
            initializer="zeros",
        )

    def reset_state(self):
        self.true_positive_buckets.assign(tf.zeros_like(self.true_positive_buckets))
        self.false_positive_buckets.assign(tf.zeros_like(self.false_positive_buckets))
        self.ground_truths.assign(tf.zeros_like(self.ground_truths))

    @tf.function()
    def update_state(self, y_true, y_pred, sample_weight=None):
        num_images = tf.shape(y_true)[0]

        if sample_weight is not None:
            raise ValueError("Received unsupported `sample_weight` to `update_state`")

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

            detections = detections
            if self.max_detections < tf.shape(detections)[0]:
                detections = detections[: self.max_detections]

            true_positives_update = tf.TensorArray(
                tf.int32, size=self.num_class_ids * self.num_iou_thresholds
            )
            false_positives_update = tf.TensorArray(
                tf.int32, size=self.num_class_ids * self.num_iou_thresholds
            )
            ground_truths_update = tf.TensorArray(tf.int32, size=self.num_class_ids)

            for c_i in range(self.num_class_ids):
                category_id = self.class_ids[c_i]
                ground_truths = utils.filter_boxes(
                    ground_truths, value=category_id, axis=bbox.CLASS
                )

                detections = utils.filter_boxes(
                    detections, value=category_id, axis=bbox.CLASS
                )
                if self.max_detections < tf.shape(detections)[0]:
                    detections = detections[: self.max_detections]

                ground_truths_update = ground_truths_update.write(
                    c_i, tf.shape(ground_truths)[0]
                )

                ious = iou_lib.compute_ious_for_image(ground_truths, detections)

                for iou_i in range(self.num_iou_thresholds):
                    iou_threshold = self.iou_thresholds[iou_i]
                    pred_matches = utils.match_boxes(ious, iou_threshold)

                    dt_scores = detections[:, bbox.CONFIDENCE]

                    tps = pred_matches != -1
                    fps = pred_matches == -1

                    confidence_buckets = tf.cast(
                        tf.math.floor(self.num_buckets * dt_scores), tf.int32
                    )

                    tps_by_bucket = tf.gather_nd(
                        confidence_buckets, indices=tf.where(tps)
                    )
                    fps_by_bucket = tf.gather_nd(
                        confidence_buckets, indices=tf.where(fps)
                    )

                    tp_counts_per_bucket = tf.math.bincount(
                        tps_by_bucket,
                        minlength=self.num_buckets,
                        maxlength=self.num_buckets,
                    )
                    fp_counts_per_bucket = tf.math.bincount(
                        fps_by_bucket,
                        minlength=self.num_buckets,
                        maxlength=self.num_buckets,
                    )
                    true_positives_update = true_positives_update.write(
                        (self.num_iou_thresholds * c_i) + iou_i, tp_counts_per_bucket
                    )
                    false_positives_update = false_positives_update.write(
                        (self.num_iou_thresholds * c_i) + iou_i, fp_counts_per_bucket
                    )

            true_positives_update = tf.reshape(
                true_positives_update.stack(),
                (self.num_class_ids, self.num_iou_thresholds, self.num_buckets),
            )
            false_positives_update = tf.reshape(
                false_positives_update.stack(),
                (self.num_class_ids, self.num_iou_thresholds, self.num_buckets),
            )

            true_positive_buckets_update = (
                true_positive_buckets_update + true_positives_update
            )
            false_positive_buckets_update = (
                false_positive_buckets_update + false_positives_update
            )
            ground_truth_boxes_update = (
                ground_truth_boxes_update + ground_truths_update.stack()
            )

        self.ground_truths.assign_add(ground_truth_boxes_update)
        self.true_positive_buckets.assign_add(true_positive_buckets_update)
        self.false_positive_buckets.assign_add(false_positive_buckets_update)

    @tf.function()
    def result(self):
        true_positives = tf.cast(self.true_positive_buckets, self.dtype)
        false_positives = tf.cast(self.false_positive_buckets, self.dtype)
        ground_truths = tf.cast(self.ground_truths, self.dtype)

        true_positives_sum = tf.cumsum(true_positives, axis=-1)
        false_positives_sum = tf.cumsum(false_positives, axis=-1)

        present_categories = tf.math.reduce_sum(tf.cast(ground_truths != 0, tf.int32))

        if present_categories == 0:
            return 0.0

        # tp_sum shape, [categories, iou_thr, n_buckets]
        recalls = tf.math.divide_no_nan(
            true_positives_sum, ground_truths[:, None, None]
        )
        precisions = tf.math.divide_no_nan(
            true_positives_sum, (false_positives_sum + true_positives_sum)
        )

        result = tf.TensorArray(
            tf.float32, size=self.num_class_ids * self.num_iou_thresholds
        )

        # so in this case this should be: [1, 1]
        for i in range(self.num_class_ids):
            for j in range(self.num_iou_thresholds):
                recalls_i = recalls[i, j]
                precisions_i = precisions[i, j]
                inds = tf.searchsorted(
                    recalls_i, tf.constant(self.recall_thresholds), side="left"
                )

                precision_result = tf.TensorArray(
                    self.dtype, size=len(self.recall_thresholds)
                )

                # TODO(lukewood): Vectorize this, this should be trivial with
                # gather operations.
                for r_i in tf.range(len(self.recall_thresholds)):
                    p_i = inds[r_i]
                    if p_i < tf.shape(precisions_i)[0]:
                        result_for_threshold = precisions_i[p_i]
                        precision_result = precision_result.write(
                            r_i, result_for_threshold
                        )

                precision_per_recall_threshold = precision_result.stack()
                tf.print(precision_per_recall_threshold)
                result_ij = tf.math.reduce_mean(precision_per_recall_threshold, axis=-1)
                result = result.write(j + i * self.num_iou_thresholds, result_ij)

        result = tf.reshape(
            result.stack(), (self.num_class_ids, self.num_iou_thresholds)
        )
        result = tf.math.reduce_mean(result, axis=-1)
        result = tf.math.reduce_sum(result, axis=0) / tf.cast(
            present_categories, tf.float32
        )
        return result
