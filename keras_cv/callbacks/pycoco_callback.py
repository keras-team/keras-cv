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

from keras_cv import bounding_box
from keras_cv.metrics.coco import compute_pycoco_metrics


class PyCOCOCallback(Callback):
    def __init__(self, validation_data, bounding_box_format, **kwargs):
        self.model = None
        self.val_data = validation_data
        self.bounding_box_format = bounding_box_format
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        gt = {}
        preds = {}

        for i, batch in enumerate(self.val_data):
            gt_i, preds_i = self._eval_batch(batch, i)

            for k, v in six.iteritems(preds_i):
                if k not in preds:
                    preds[k] = [v]
                else:
                    preds[k].append(v)

            for k, v in six.iteritems(gt_i):
                if k not in gt:
                    gt[k] = [v]
                else:
                    gt[k].append(v)

        metrics = compute_pycoco_metrics(gt, preds)
        # Mark these as validation metrics by prepending a val_ prefix
        metrics = {"val_" + name: val for name, val in metrics.items()}

        logs.update(metrics)

    def _eval_batch(self, batch, index):
        images, y = batch
        gt_boxes = y["gt_boxes"]
        gt_classes = y["gt_classes"]
        batch_size = images.shape[0]
        height = images.shape[1]
        width = images.shape[2]

        gt_boxes = bounding_box.convert_format(
            gt_boxes, source=self.bounding_box_format, target="yxyx"
        )

        source_ids = tf.strings.join(
            [
                tf.strings.as_string(tf.tile(tf.constant([index]), [batch_size])),
                tf.strings.as_string(
                    tf.linspace(1, batch_size, batch_size), precision=0
                ),
            ],
            separator="/",
        )

        ground_truth = {}
        ground_truth["source_id"] = source_ids
        ground_truth["height"] = tf.tile(tf.constant([height]), [batch_size])
        ground_truth["width"] = tf.tile(tf.constant([width]), [batch_size])

        num_dets = gt_classes.get_shape().as_list()[1]
        ground_truth["num_detections"] = tf.tile(tf.constant([num_dets]), [batch_size])
        ground_truth["boxes"] = gt_boxes
        ground_truth["classes"] = gt_classes

        y_pred = self.model.predict(images)
        y_pred = bounding_box.convert_format(
            y_pred, source=self.bounding_box_format, target="yxyx"
        )

        predictions = {}
        predictions["num_detections"] = y_pred.row_lengths()
        y_pred = y_pred.to_tensor(-1)

        predictions["source_id"] = source_ids
        predictions["detection_boxes"] = y_pred[:, :, :4]
        predictions["detection_classes"] = y_pred[:, :, 4]
        predictions["detection_scores"] = y_pred[:, :, 5]

        return ground_truth, predictions
