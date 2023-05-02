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

from keras_cv import bounding_box
from keras_cv.metrics.coco import compute_pycoco_metrics
from keras_cv.models.object_detection.__internal__ import unpack_input
from keras_cv.utils.conditional_imports import assert_pycocotools_installed


class PyCOCOCallback(Callback):
    def __init__(
        self, validation_data, bounding_box_format, cache=True, **kwargs
    ):
        """Creates a callback to evaluate PyCOCO metrics on a validation
        dataset.

        Args:
            validation_data: a tf.data.Dataset containing validation data.
                Entries should have the form ```(images, {"boxes": boxes,
                "classes": classes})```.
            bounding_box_format: the KerasCV bounding box format used in the
                validation dataset (e.g. "xywh")
            cache: whether the callback should cache the dataset between
                iterations. Note that if the validation dataset has shuffling of
                any kind (e.g. from `shuffle_files=True` in a call to TFDS).
                Load or a call to tf.data.Dataset.shuffle() with
                `reshuffle_each_iteration=True`), you **must** cache the dataset
                to preserve iteration order. This will store your entire dataset
                in main memory, so for large datasets consider avoiding shuffle
                operations and passing `cache=False`.
        """
        assert_pycocotools_installed("PyCOCOCallback")
        self.model = None
        self.val_data = validation_data
        if cache:
            # We cache the dataset to preserve a consistent iteration order.
            self.val_data = self.val_data.cache()
        self.bounding_box_format = bounding_box_format
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def images_only(data):
            images, boxes = unpack_input(data)
            return images

        def boxes_only(data):
            images, boxes = unpack_input(data)
            return bounding_box.to_ragged(boxes)

        images_only_ds = self.val_data.map(images_only)
        y_pred = self.model.predict(images_only_ds)
        box_pred = y_pred["boxes"]
        cls_pred = y_pred["classes"]
        confidence_pred = y_pred["confidence"]
        valid_det = y_pred["num_detections"]

        gt = [boxes for boxes in self.val_data.map(boxes_only)]
        gt_boxes = tf.concat(
            [boxes["boxes"] for boxes in gt],
            axis=0,
        )
        gt_classes = tf.concat(
            [boxes["classes"] for boxes in gt],
            axis=0,
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

        ground_truth = {
            "source_id": [source_ids],
            "height": [tf.tile(tf.constant([height]), [total_images])],
            "width": [tf.tile(tf.constant([width]), [total_images])],
            "num_detections": [gt_boxes.row_lengths(axis=1)],
            "boxes": [gt_boxes.to_tensor(-1)],
            "classes": [gt_classes.to_tensor(-1)],
        }

        box_pred = bounding_box.convert_format(
            box_pred, source=self.bounding_box_format, target="yxyx"
        )

        predictions = {
            "source_id": [source_ids],
            "detection_boxes": [box_pred],
            "detection_classes": [cls_pred],
            "detection_scores": [confidence_pred],
            "num_detections": [valid_det],
        }

        metrics = compute_pycoco_metrics(ground_truth, predictions)
        # Mark these as validation metrics by prepending a val_ prefix
        metrics = {"val_" + name: val for name, val in metrics.items()}

        logs.update(metrics)
