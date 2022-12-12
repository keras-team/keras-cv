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
import six
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
                should have the form ```{"point_clouds": point_clouds,
                "bounding_boxes": bounding_boxes}```.
            config: an optional `metrics_pb2.Config` object from WOD to specify
                what metrics should be evaluated.
        """
        self.model = None
        self.val_data = validation_data
        self.evaluator = WODDetectionEvaluator(config=config)
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        frame_id = 0
        for batch in self.val_data:
            gt_i, preds_i, batch_size = self._eval_batch(batch, frame_id)
            frame_id += batch_size
            evaluator.update_state(gt_i, preds_i)

        metrics = evaluator.evaluate()

        logs.update(metrics)

    def _eval_batch(self, batch, frame_id):
        point_clouds = batch["point_clouds"]
        boxes = batch["bounding_boxes"]
        batch_size = boxes.shape[0]

        frame_ids = tf.linspace(frame_id, frame_id + batch_size - 1, batch_size)

        ground_truth = {}
        ground_truth["ground_truth_frame_id"] = frame_ids
        ground_truth["ground_truth_bbox"] = boxes[:, :: CENTER_XYZ_DXDYDZ_PHI.DZ + 1]
        ground_truth["ground_truth_type"] = boxes[:, :, CENTER_XYZ_DXDYDZ_PHI.CLASS]
        ground_truth["ground_truth_difficulty"] = boxes[
            :, :, CENTER_XYZ_DXDYDZ_PHI.CLASS + 1
        ]

        y_pred = self.model.predict_on_batch(point_clouds)

        predictions = {}

        predictions["prediction_frame_id"] = source_ids
        predictions["prediction_bbox"] = y_pred[:, :, : CENTER_XYZ_DXDYDZ_PHI.DZ + 1]
        predictions["prediction_type"] = y_pred[:, :, CENTER_XYZ_DXDYDZ_PHI.CLASS]
        predictions["prediction_score"] = y_pred[:, :, CENTER_XYZ_DXDYDZ_PHI.CLASS + 1]
        predictions["prediction_overlap_nlz"] = tf.zeros(y_pred.shape[:-1])

        return ground_truth, predictions
