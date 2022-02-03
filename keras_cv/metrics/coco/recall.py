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

from keras_cv.metrics.coco import iou as iou_lib
from keras_cv.metrics.coco import utils
from keras_cv.utils import bbox


class COCORecall(keras.metrics.Metric):
    """COCORecall computes the COCO recall metric.

    Args:
        iou_thresholds: defaults to [0.5:0.05:0.95].
        category_ids: no default, users must provide.
        area_range: area range to consider bounding boxes in. Defaults to all.
        max_detections: number of maximum detections a model is allowed to make.

    Usage:
        COCORecall accepts two Tensors as input to it's `update_state` method.
        These Tensors represent bounding boxes in `corners` format.  Utilities
        to convert Tensors from `xywh` to `corners` format can be found in
        `keras_cv.utils.bbox`.

        Each image in a dataset may have a different number of bounding boxes,
        both in the ground truth dataset and the prediction set.  In order to
        account for this, users must pad Tensors with `-1`s to indicate unused
        boxes.  A utility function to perform this padding is available at
        `keras_cv_.utils.bbox.pad_bbox_batch_to_shape`.

        ```
        coco_recall = COCORecall(
            max_detections=100,
            category_ids=[1],
            area_range=(32**2, 64**2),
        )

        y_true = np.array([[[0, 0, 10, 10, 1], [5, 5, 10, 10, 1]]]).astype(np.float32)
        y_pred = np.array([[[0, 0, 10, 10, 1, 1.0], [5, 5, 10, 10, 1, 0.9]]]).astype(
            np.float32
        )
        coco_recall.update_state(y_true, y_pred)
        coco_recall.result()
        # > 0.0
        ```
    """

    def __init__(
        self,
        category_ids,
        iou_thresholds=None,
        area_range=(0, 1e9**2),
        max_detections=100,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Initialize parameter values

        iou_thresholds = iou_thresholds or [x / 100.0 for x in range(50, 100, 5)]
        self.iou_thresholds = iou_thresholds
        self.category_ids = category_ids

        self.area_range = area_range
        self.max_detections = max_detections

        # Initialize result counters
        num_thresholds = len(iou_thresholds)
        num_categories = len(category_ids)

        self.true_positives = self.add_weight(
            name="true_positives",
            shape=(num_thresholds, num_categories),
            dtype=tf.float32,
            initializer=initializers.Zeros(),
        )
        self.ground_truth_boxes = self.add_weight(
            name="ground_truth_boxes",
            shape=(num_categories,),
            dtype=tf.float32,
            initializer=initializers.Zeros(),
        )

    def reset_state(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.ground_truth_boxes.assign(tf.zeros_like(self.ground_truth_boxes))

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Args:
            y_true: a bounding box Tensor in corners format.
            y_pred: a bounding box Tensor in corners format.
            sample_weight: Currently unsupported.
        """
        if sample_weight:
            raise NotImplementedError(
                "sample_weight is not yet supported in keras_cv COCO metrics."
            )
        num_images = tf.shape(y_true)[0]

        iou_thresholds = tf.constant(self.iou_thresholds, dtype=tf.float32)
        category_ids = tf.constant(self.category_ids, dtype=tf.float32)

        num_thresholds = tf.shape(iou_thresholds)[0]
        num_categories = tf.shape(category_ids)[0]

        # Sort by bbox.CONFIDENCE to make maxDetections easy to compute.
        true_positives_update = tf.zeros_like(self.true_positives)
        ground_truth_boxes_update = tf.zeros_like(self.ground_truth_boxes)

        for img in tf.range(num_images):
            sentinel_filtered_y_true = utils.filter_out_sentinels(y_true[img])
            sentinel_filtered_y_pred = utils.filter_out_sentinels(y_pred[img])

            area_filtered_y_true = utils.filter_boxes_by_area_range(
                sentinel_filtered_y_true, self.area_range[0], self.area_range[1]
            )
            # TODO(lukewood): try filtering area after max dts.
            area_filtered_y_pred = utils.filter_boxes_by_area_range(
                sentinel_filtered_y_pred, self.area_range[0], self.area_range[1]
            )

            for k_i in tf.range(num_categories):
                category = category_ids[k_i]

                category_filtered_y_pred = utils.filter_boxes(
                    area_filtered_y_pred, value=category, axis=bbox.CLASS
                )

                detections = category_filtered_y_pred
                if self.max_detections < tf.shape(category_filtered_y_pred)[0]:
                    detections = category_filtered_y_pred[: self.max_detections]

                ground_truths = utils.filter_boxes(
                    area_filtered_y_true, value=category, axis=bbox.CLASS
                )

                ious = iou_lib.compute_ious_for_image(ground_truths, detections)

                for t_i in tf.range(num_thresholds):
                    threshold = iou_thresholds[t_i]
                    pred_matches = utils.match_boxes(
                        ground_truths, detections, threshold, ious
                    )

                    indices = [t_i, k_i]
                    true_positives = tf.cast(pred_matches != -1, tf.float32)
                    true_positives_sum = tf.math.reduce_sum(true_positives, axis=-1)

                    true_positives_update = tf.tensor_scatter_nd_add(
                        true_positives_update, [indices], [true_positives_sum]
                    )

                ground_truth_boxes_update = tf.tensor_scatter_nd_add(
                    ground_truth_boxes_update,
                    [[k_i]],
                    [tf.cast(tf.shape(ground_truths)[0], tf.float32)],
                )

        self.true_positives.assign_add(true_positives_update)
        self.ground_truth_boxes.assign_add(ground_truth_boxes_update)

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
