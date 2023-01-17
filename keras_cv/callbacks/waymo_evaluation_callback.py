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
from keras.callbacks import Callback
from waymo_open_dataset.metrics.python.wod_detection_evaluator import (
    WODDetectionEvaluator,
)

from keras_cv.bounding_box_3d import CENTER_XYZ_DXDYDZ_PHI


class WaymoEvaluationCallback(Callback):
    def __init__(self, validation_data, config=None, **kwargs):
        """Creates a callback to evaluate Waymo Open Dataset (WOD) metrics on a
        validation dataset.

        Args:
            validation_data: a tf.data.Dataset containing validation data. Entries
                should have the form `(point_clouds, {"bounding_boxes": bounding_boxes}`.
                Padded bounding box should have a class of -1 to be correctly filtered out.
            config: an optional `metrics_pb2.Config` object from WOD to specify
                what metrics should be evaluated.
        """
        self.model = None
        self.val_data = validation_data
        self.evaluator = WODDetectionEvaluator(config=config)
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        gt, preds = self._eval_dataset(self.val_data)
        self.evaluator.update_state(gt, preds)

        metrics = self.evaluator.evaluate()

        metrics_dict = {
            "average_precision": metrics.average_precision,
            "average_precision_ha_weighted": metrics.average_precision_ha_weighted,
            "precision_recall": metrics.precision_recall,
            "precision_recall_ha_weighted": metrics.precision_recall_ha_weighted,
            "breakdown": metrics.breakdown,
        }

        logs.update(metrics_dict)

    def _eval_dataset(self, dataset):
        def point_clouds_only(point_clouds, target):
            return point_clouds

        def boxes_only(point_clouds, target):
            return target["boxes"]

        predicted_boxes, predicted_classes = self.model.predict(
            dataset.map(point_clouds_only)
        )
        gt_boxes = tf.concat([x for x in iter(dataset.map(boxes_only))], axis=0)

        boxes_per_gt_frame = gt_boxes.shape[1]
        num_frames = gt_boxes.shape[0]

        gt_boxes = tf.reshape(gt_boxes, (num_frames * boxes_per_gt_frame, 9))

        # Remove boxes with class of -1 (these are non-boxes that come from padding)
        gt_real_boxes = tf.not_equal(gt_boxes[:, CENTER_XYZ_DXDYDZ_PHI.CLASS], -1)
        gt_boxes = tf.boolean_mask(gt_boxes, gt_real_boxes)

        frame_ids = tf.cast(tf.linspace(1, num_frames, num_frames), tf.int64)

        ground_truth = {}
        ground_truth["ground_truth_frame_id"] = tf.boolean_mask(
            tf.repeat(frame_ids, boxes_per_gt_frame), gt_real_boxes
        )
        ground_truth["ground_truth_bbox"] = gt_boxes[:, : CENTER_XYZ_DXDYDZ_PHI.PHI + 1]
        ground_truth["ground_truth_type"] = tf.cast(
            gt_boxes[:, CENTER_XYZ_DXDYDZ_PHI.CLASS], tf.uint8
        )
        ground_truth["ground_truth_difficulty"] = tf.cast(
            gt_boxes[:, CENTER_XYZ_DXDYDZ_PHI.CLASS + 1], tf.uint8
        )

        boxes_per_pred_frame = predicted_boxes.shape[1]
        total_predicted_boxes = boxes_per_pred_frame * num_frames
        predicted_boxes = tf.reshape(predicted_boxes, (total_predicted_boxes, 7))
        predicted_classes = tf.reshape(predicted_classes, (total_predicted_boxes, 2))
        # Remove boxes with class of -1 (these are non-boxes that come from padding)
        pred_real_boxes = tf.reduce_all(predicted_classes != -1, axis=[-1])
        predicted_boxes = tf.boolean_mask(predicted_boxes, pred_real_boxes)
        predicted_classes = tf.boolean_mask(predicted_classes, pred_real_boxes)

        predictions = {}

        predictions["prediction_frame_id"] = tf.boolean_mask(
            tf.repeat(frame_ids, boxes_per_pred_frame), pred_real_boxes
        )
        predictions["prediction_bbox"] = predicted_boxes
        predictions["prediction_type"] = tf.cast(
            tf.argmax(predicted_classes, axis=-1), tf.uint8
        )
        predictions["prediction_score"] = tf.reduce_max(predicted_classes, axis=-1)
        predictions["prediction_overlap_nlz"] = tf.cast(
            tf.zeros(predicted_boxes.shape[0]), tf.bool
        )

        return ground_truth, predictions
