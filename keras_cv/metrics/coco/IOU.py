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


class COCOIOU(tf.keras.metrics.Metric):
    """Computes the Intersection-Over-Union metric for specific target classes.
    General definition and computation:
    Intersection-Over-Union is a common evaluation metric for semantic image
    segmentation.
    For an individual class, the IoU metric is defined as follows:
    ```
    iou = true_positives / (true_positives + false_positives + false_negatives)
    ```
    To compute IoUs, the predictions are accumulated in a confusion matrix,
    weighted by `sample_weight` and the metric is then calculated from it.
    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.
    Note, this class first computes IoUs for all individual classes, then
    returns the mean of IoUs for the classes that are specified by
    `target_class_ids`. If `target_class_ids` has only one id value, the IoU of
    that specific class is returned
    Args:
        class_ids: The class IDs to evaluate the metric for.  To evaluate for
            all classes in over a set of sequentially labelled classes, pass
            `range(classes)`.
        bounding_box_format: Format of the incoming bounding boxes.  Supported values
            "xyxy".
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

     ```python
        coco_map = keras_cv.metrics.COCOIOU(
        bounding_box_format='xyxy',
        max_detections=100,
        class_ids=[1]
    )
    coco_map.update_state([0, 0, 1, 1], [0, 1, 0, 1])
    coco_map.result()
    #
    ```
    """

    def __init__(
        self,
        class_ids,
        bounding_box_format,
        recall_thresholds=None,
        area_range=None,
        max_detections=100,
        num_buckets=10000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Initialize parameter values
        self.bounding_box_format = bounding_box_format
        self.area_range = area_range
        self.max_detections = max_detections
        self.class_ids = list(class_ids)
        self.recall_thresholds = recall_thresholds or [x / 100 for x in range(0, 101)]
        self.num_buckets = num_buckets

        self.num_class_ids = len(self.class_ids)

        if any([c < 0 for c in class_ids]):
            raise ValueError(
                "class_ids must be positive.  Got " f"class_ids={class_ids}"
            )

        self.y_true = self.add_weight(
            "y_true",
            shape=(self.num_class_ids,),
            dtype=tf.int32,
            initializer="zeros",
        )
        self.y_pred = self.add_weight(
            "Y_pred",
            shape=(
                self.num_class_ids,
                self.num_iou_thresholds,
                self.num_buckets,
            ),
            dtype=tf.int32,
            initializer="zeros",
        )


    def reset_state(self):
        self.y_true_buckets.assign(tf.zeros_like(self.y_true_buckets))
        self.y_pred.assign(tf.zeros_like(self.y_pred))


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



    @tf.function()
    def result(self):
        """Compute the intersection-over-union via the confusion matrix."""
        sum_over_row = tf.cast(
            tf.reduce_sum(self.total_cm, axis=0), dtype=self._dtype
        )
        sum_over_col = tf.cast(
            tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype
        )
        true_positives = tf.cast(
            tf.linalg.tensor_diag_part(self.total_cm), dtype=self._dtype
        )

        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col - true_positives

        # Only keep the target classes
        true_positives = tf.gather(true_positives, self.target_class_ids)
        denominator = tf.gather(denominator, self.target_class_ids)

        # If the denominator is 0, we need to ignore the class.
        num_valid_entries = tf.reduce_sum(
            tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype)
        )

        iou = tf.math.divide_no_nan(true_positives, denominator)

        return tf.math.divide_no_nan(
            tf.reduce_sum(iou, name="mean_iou"), num_valid_entries
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "class_ids": self.class_ids,
                "bounding_box_format": self.bounding_box_format,
                "recall_thresholds": self.recall_thresholds,
                "area_range": self.area_range,
                "max_detections": self.max_detections,
                "num_buckets": self.num_buckets,
            }
        )
        return config
