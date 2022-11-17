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

        def images_only(images, boxes):
            return images

        def boxes_only(images, boxes):
            return boxes

        images_only_ds = self.val_data.map(images_only)
        y_pred = self.model.predict(images_only_ds)

        gt = [boxes for boxes in self.val_data.map(boxes_only)]
        gt_boxes = tf.concat(
            [tf.RaggedTensor.from_tensor(boxes["gt_boxes"]) for boxes in gt], axis=0
        )
        gt_classes = tf.concat(
            [tf.RaggedTensor.from_tensor(boxes["gt_classes"]) for boxes in gt], axis=0
        )

        first_image_batch = next(iter(images_only_ds))
        height = first_image_batch.shape[1]
        width = first_image_batch.shape[2]
        total_images = gt_boxes.shape[0]

        gt_boxes = bounding_box.convert_format(
            gt_boxes, source=self.bounding_box_format, target="yxyx"
        )

        source_ids = tf.strings.as_string(
            tf.linspace(1, total_images, total_images), precision=0
        )

        ground_truth = {}
        ground_truth["source_id"] = [source_ids]
        ground_truth["height"] = [tf.tile(tf.constant([height]), [total_images])]
        ground_truth["width"] = [tf.tile(tf.constant([width]), [total_images])]

        num_dets = gt_classes.get_shape().as_list()[1]
        ground_truth["num_detections"] = [gt_boxes.row_lengths(axis=1)]
        ground_truth["boxes"] = [gt_boxes.to_tensor(-1)]
        ground_truth["classes"] = [gt_classes.to_tensor(-1)]

        y_pred = bounding_box.convert_format(
            y_pred, source=self.bounding_box_format, target="yxyx"
        )

        predictions = {}
        predictions["num_detections"] = [y_pred.row_lengths()]
        y_pred = y_pred.to_tensor(-1)

        predictions["source_id"] = [source_ids]
        predictions["detection_boxes"] = [y_pred[:, :, :4]]
        predictions["detection_classes"] = [y_pred[:, :, 4]]
        predictions["detection_scores"] = [y_pred[:, :, 5]]

        metrics = compute_pycoco_metrics(ground_truth, predictions)
        # Mark these as validation metrics by prepending a val_ prefix
        metrics = {"val_" + name: val for name, val in metrics.items()}

        logs.update(metrics)
