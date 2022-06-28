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
import warnings

import tensorflow as tf

from keras_cv import bounding_box
from keras_cv.metrics.coco import utils
from keras_cv.utils import iou as iou_lib


class COCOMeanAveragePrecision(tf.keras.metrics.Metric):
    """COCOMeanAveragePrecision computes an approximation of MaP.

    Args:
        class_ids: The class IDs to evaluate the metric for.  To evaluate for
            all classes in over a set of sequentially labelled classes, pass
            `range(num_classes)`.
        bounding_box_format: Format of the incoming bounding boxes.  Supported values
            are "xywh", "center_xywh", "xyxy".
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
        num_buckets: num_buckets is used to select the number of confidence
            buckets predictions are placed into.  Instead of computation MaP
            over each incrementally selected set of bounding boxes, we instead
            place them into buckets.  This makes distributed computation easier.
            Increasing buckets improves accuracy of the metric, while decreasing
            buckets improves performance.  This is a tradeoff you must weight
            for your use case.  Defaults to 10,000 which is sufficiently large
            for most use cases.

    Usage:

    COCOMeanAveragePrecision accepts two Tensors as input to it's
    `update_state()` method.  These Tensors represent bounding boxes in
    `corners` format.  Utilities to convert Tensors from `xywh` to `corners`
    format can be found in `keras_cv.utils.bounding_box`.

    Each image in a dataset may have a different number of bounding boxes,
    both in the ground truth dataset and the prediction set.  In order to
    account for this, you may either pass a `tf.RaggedTensor`, or pad Tensors
    with `-1`s to indicate unused boxes.  A utility function to perform this
    padding is available at
    `keras_cv.bounding_box.pad_batch_to_shape()`.

    ```python
    coco_map = keras_cv.metrics.COCOMeanAveragePrecision(
        bounding_box_format='xyxy',
        max_detections=100,
        class_ids=[1]
    )

    y_true = np.array([[[0, 0, 10, 10, 1], [20, 20, 10, 10, 1]]]).astype(np.float32)
    y_pred = np.array([[[0, 0, 10, 10, 1, 1.0], [5, 5, 10, 10, 1, 0.9]]]).astype(
        np.float32
    )
    coco_map.update_state(y_true, y_pred)
    coco_map.result()
    # 0.24752477
    ```
    """

    def __init__(
        self,
        class_ids,
        bounding_box_format,
        recall_thresholds=None,
        iou_thresholds=None,
        area_range=None,
        max_detections=100,
        num_buckets=10000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Initialize parameter values
        self.bounding_box_format = bounding_box_format
        self.iou_thresholds = iou_thresholds or [x / 100.0 for x in range(50, 100, 5)]
        self.area_range = area_range
        self.max_detections = max_detections
        self.class_ids = list(class_ids)
        self.recall_thresholds = recall_thresholds or [x / 100 for x in range(0, 101)]
        self.num_buckets = num_buckets

        self.num_iou_thresholds = len(self.iou_thresholds)
        self.num_class_ids = len(self.class_ids)

        if any([c < 0 for c in class_ids]):
            raise ValueError(
                "class_ids must be positive.  Got " f"class_ids={class_ids}"
            )

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
                self.num_buckets,
            ),
            dtype=tf.int32,
            initializer="zeros",
        )
        self.false_positive_buckets = self.add_weight(
            "false_positive_buckets",
            shape=(
                self.num_class_ids,
                self.num_iou_thresholds,
                self.num_buckets,
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
        if sample_weight is not None:
            warnings.warn(
                "sample_weight is not yet supported in keras_cv COCO metrics."
            )
        y_true = tf.cast(y_true, self.compute_dtype)
        y_pred = tf.cast(y_pred, self.compute_dtype)

        if isinstance(y_true, tf.RaggedTensor):
            y_true = y_true.to_tensor(default_value=-1)
        if isinstance(y_pred, tf.RaggedTensor):
            y_pred = y_pred.to_tensor(default_value=-1)

        y_true = bounding_box.convert_format(
            y_true,
            source=self.bounding_box_format,
            target="xyxy",
            dtype=self.compute_dtype,
        )
        y_pred = bounding_box.convert_format(
            y_pred,
            source=self.bounding_box_format,
            target="xyxy",
            dtype=self.compute_dtype,
        )

        class_ids = tf.constant(self.class_ids, dtype=self.compute_dtype)
        iou_thresholds = tf.constant(self.iou_thresholds, dtype=self.compute_dtype)

        num_images = tf.shape(y_true)[0]

        y_pred = utils.sort_bounding_boxes(y_pred, axis=bounding_box.XYXY.CONFIDENCE)

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

            if self.max_detections < tf.shape(detections)[0]:
                detections = detections[: self.max_detections]

            true_positives_update = tf.TensorArray(
                tf.int32, size=self.num_class_ids * self.num_iou_thresholds
            )
            false_positives_update = tf.TensorArray(
                tf.int32, size=self.num_class_ids * self.num_iou_thresholds
            )
            ground_truths_update = tf.TensorArray(tf.int32, size=self.num_class_ids)

            for c_i in tf.range(self.num_class_ids):
                category_id = class_ids[c_i]
                ground_truths_by_category = utils.filter_boxes(
                    ground_truths, value=category_id, axis=bounding_box.XYXY.CLASS
                )
                detections_by_category = utils.filter_boxes(
                    detections, value=category_id, axis=bounding_box.XYXY.CLASS
                )
                if self.max_detections < tf.shape(detections_by_category)[0]:
                    detections_by_category = detections_by_category[
                        : self.max_detections
                    ]

                ground_truths_update = ground_truths_update.write(
                    c_i, tf.shape(ground_truths_by_category)[0]
                )

                ious = iou_lib.compute_ious_for_image(
                    ground_truths_by_category, detections_by_category
                )

                for iou_i in tf.range(self.num_iou_thresholds):
                    iou_threshold = iou_thresholds[iou_i]
                    pred_matches = utils.match_boxes(ious, iou_threshold)

                    dt_scores = detections_by_category[:, bounding_box.XYXY.CONFIDENCE]

                    true_positives = pred_matches != -1
                    false_positives = pred_matches == -1

                    dt_scores_clipped = tf.clip_by_value(dt_scores, 0.0, 1.0)
                    # We must divide by 1.01 to prevent off by one errors.
                    confidence_buckets = tf.cast(
                        tf.math.floor(self.num_buckets * (dt_scores_clipped / 1.01)),
                        tf.int32,
                    )
                    true_positives_by_bucket = tf.gather_nd(
                        confidence_buckets, indices=tf.where(true_positives)
                    )
                    false_positives_by_bucket = tf.gather_nd(
                        confidence_buckets, indices=tf.where(false_positives)
                    )

                    true_positive_counts_per_bucket = tf.math.bincount(
                        true_positives_by_bucket,
                        minlength=self.num_buckets,
                        maxlength=self.num_buckets,
                    )
                    false_positives_counts_per_bucket = tf.math.bincount(
                        false_positives_by_bucket,
                        minlength=self.num_buckets,
                        maxlength=self.num_buckets,
                    )

                    true_positives_update = true_positives_update.write(
                        (self.num_iou_thresholds * c_i) + iou_i,
                        true_positive_counts_per_bucket,
                    )
                    false_positives_update = false_positives_update.write(
                        (self.num_iou_thresholds * c_i) + iou_i,
                        false_positives_counts_per_bucket,
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

        recalls = tf.math.divide_no_nan(
            true_positives_sum, ground_truths[:, None, None]
        )
        precisions = true_positives_sum / (false_positives_sum + true_positives_sum)

        result = tf.TensorArray(
            tf.float32, size=self.num_class_ids * self.num_iou_thresholds
        )
        zero_pad = tf.zeros(shape=(1,), dtype=tf.float32)
        # so in this case this should be: [1, 1]
        for i in tf.range(self.num_class_ids):
            for j in tf.range(self.num_iou_thresholds):
                recalls_i = recalls[i, j]
                precisions_i = precisions[i, j]

                # recall threshold=0 finds the first bucket always
                # this is different from the original implementation because the
                # original implementation always has at least one bounding box
                # in the first bucket.
                #
                # as such, we need to mask out the buckets where there is at
                # least one bounding box  Therefore, we must filter out the
                # buckets where (precisions_i) is NaN, as that implies a divide
                # by zero.

                inds = tf.where(not tf.math.is_nan(precisions_i))
                recalls_i = tf.gather_nd(recalls_i, inds)
                precisions_i = tf.gather_nd(precisions_i, inds)

                inds = tf.searchsorted(
                    recalls_i, tf.constant(self.recall_thresholds), side="left"
                )

                # if searchsorted returns len(precisions)+1, we should return 0
                precisions_i = tf.concat([precisions_i, zero_pad], axis=-1)
                precision_per_recall_threshold = tf.gather(precisions_i, inds)

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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "class_ids": self.class_ids,
                "bounding_box_format": self.bounding_box_format,
                "recall_thresholds": self.recall_thresholds,
                "iou_thresholds": self.iou_thresholds,
                "area_range": self.area_range,
                "max_detections": self.max_detections,
                "num_buckets": self.num_buckets,
            }
        )
        return config
