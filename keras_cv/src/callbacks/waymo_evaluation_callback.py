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
from tensorflow.keras.callbacks import Callback

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.bounding_box_3d import CENTER_XYZ_DXDYDZ_PHI
from keras_cv.src.utils import assert_waymo_open_dataset_installed

try:
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.metrics.python.wod_detection_evaluator import (
        WODDetectionEvaluator,
    )
    from waymo_open_dataset.protos import breakdown_pb2
    from waymo_open_dataset.protos import metrics_pb2
except ImportError:
    WODDetectionEvaluator = None


@keras_cv_export("keras_cv.callbacks.WaymoEvaluationCallback")
class WaymoEvaluationCallback(Callback):
    def __init__(self, validation_data, config=None, **kwargs):
        """Creates a callback to evaluate Waymo Open Dataset (WOD) metrics on a
        validation dataset.

        Args:
            validation_data: a tf.data.Dataset containing validation data.
                Entries should have the form `(point_clouds, {"bounding_boxes":
                bounding_boxes}`. Padded bounding box should have a class of -1
                to be correctly filtered out.
            config: an optional `metrics_pb2.Config` object from WOD to specify
                what metrics should be evaluated.
        """
        assert_waymo_open_dataset_installed(
            "keras_cv.callbacks.WaymoEvaluationCallback()"
        )
        self.val_data = validation_data
        self.evaluator = WODDetectionEvaluator(
            config=config or self._get_default_config()
        )
        super().__init__(**kwargs)

    def _get_default_config(self):
        """Returns the default Config proto for detection."""
        config = metrics_pb2.Config()

        config.breakdown_generator_ids.append(
            breakdown_pb2.Breakdown.OBJECT_TYPE
        )
        difficulty = config.difficulties.add()
        difficulty.levels.append(label_pb2.Label.LEVEL_1)
        difficulty.levels.append(label_pb2.Label.LEVEL_2)

        config.matcher_type = metrics_pb2.MatcherProto.TYPE_HUNGARIAN
        config.iou_thresholds.append(0.0)  # Unknown
        config.iou_thresholds.append(0.7)  # Vehicle
        config.iou_thresholds.append(0.5)  # Pedestrian
        config.iou_thresholds.append(0.5)  # Sign
        config.iou_thresholds.append(0.5)  # Cyclist
        config.box_type = label_pb2.Label.Box.TYPE_3D

        for i in range(100):
            config.score_cutoffs.append(i * 0.01)
        config.score_cutoffs.append(1.0)

        return config

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        gt, preds = self._eval_dataset(self.val_data)
        self.evaluator.update_state(gt, preds)

        metrics = self.evaluator.result()

        metrics_dict = {
            "average_precision_vehicle_l1": metrics.average_precision[0],
            "average_precision_vehicle_l2": metrics.average_precision[1],
            "average_precision_ped_l1": metrics.average_precision[2],
            "average_precision_ped_l2": metrics.average_precision[3],
        }

        logs.update(metrics_dict)

    def _eval_dataset(self, dataset):
        def point_clouds_only(point_clouds, target):
            return point_clouds

        def boxes_only(point_clouds, target):
            return target["3d_boxes"]

        model_outputs = self.model.predict(dataset.map(point_clouds_only))[
            "3d_boxes"
        ]

        def flatten_target(boxes):
            return tf.concat(
                [
                    boxes["boxes"],
                    tf.expand_dims(
                        tf.cast(boxes["classes"], tf.float32), axis=-1
                    ),
                    tf.expand_dims(
                        tf.cast(boxes["difficulty"], tf.float32), axis=-1
                    ),
                ],
                axis=-1,
            )

        gt_boxes = tf.concat(
            [flatten_target(x) for x in iter(dataset.map(boxes_only))], axis=0
        )

        boxes_per_gt_frame = gt_boxes.shape[1]
        num_frames = gt_boxes.shape[0]

        gt_boxes = tf.reshape(gt_boxes, (num_frames * boxes_per_gt_frame, 9))

        # Remove padded boxes
        gt_real_boxes = tf.concat(
            [x["mask"] for x in iter(dataset.map(boxes_only))], axis=0
        )
        gt_real_boxes = tf.reshape(
            gt_real_boxes, (num_frames * boxes_per_gt_frame)
        )
        gt_boxes = tf.boolean_mask(gt_boxes, gt_real_boxes)

        frame_ids = tf.cast(tf.linspace(1, num_frames, num_frames), tf.int64)

        ground_truth = {
            "ground_truth_frame_id": tf.boolean_mask(
                tf.repeat(frame_ids, boxes_per_gt_frame), gt_real_boxes
            ),
            "ground_truth_bbox": gt_boxes[:, : CENTER_XYZ_DXDYDZ_PHI.PHI + 1],
            "ground_truth_type": tf.cast(
                gt_boxes[:, CENTER_XYZ_DXDYDZ_PHI.CLASS], tf.uint8
            ),
            "ground_truth_difficulty": tf.cast(
                gt_boxes[:, CENTER_XYZ_DXDYDZ_PHI.CLASS + 1], tf.uint8
            ),
        }

        boxes_per_pred_frame = model_outputs["boxes"].shape[1]
        total_predicted_boxes = boxes_per_pred_frame * num_frames
        predicted_boxes = tf.reshape(
            model_outputs["boxes"], (total_predicted_boxes, 7)
        )
        predicted_classes = tf.cast(
            tf.reshape(model_outputs["classes"], (total_predicted_boxes, 1)),
            tf.uint8,
        )
        prediction_scores = tf.reshape(
            model_outputs["confidence"], (total_predicted_boxes, 1)
        )
        # Remove boxes that come from padding
        pred_real_boxes = tf.squeeze(prediction_scores > 0)
        predicted_boxes = tf.boolean_mask(predicted_boxes, pred_real_boxes)
        predicted_classes = tf.boolean_mask(predicted_classes, pred_real_boxes)
        prediction_scores = tf.boolean_mask(prediction_scores, pred_real_boxes)

        predictions = {
            "prediction_frame_id": tf.boolean_mask(
                tf.repeat(frame_ids, boxes_per_pred_frame), pred_real_boxes
            ),
            "prediction_bbox": predicted_boxes,
            "prediction_type": tf.squeeze(predicted_classes),
            "prediction_score": tf.squeeze(prediction_scores),
            "prediction_overlap_nlz": tf.cast(
                tf.zeros(predicted_boxes.shape[0]), tf.bool
            ),
        }

        return ground_truth, predictions
