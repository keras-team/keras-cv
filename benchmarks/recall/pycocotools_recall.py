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

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.initializers as initializers

from keras_cv import bounding_box
from keras_cv.bounding_box import iou as iou_lib
import utils_pymetric as utils
from object_detection_py_metric import ODPyMetric
from keras_cv.metrics.coco import compute_pycoco_metrics

import os, sys

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.initializers as initializers

from keras_cv import bounding_box
from keras_cv.bounding_box import iou as iou_lib
import utils_pymetric as utils
from object_detection_py_metric import ODPyMetric


class HidePrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class COCOToolsRecall(ODPyMetric):
    def __init__(self, metric_key):
        super().__init__()
        self.gt_boxes = []
        self.pred_boxes = []
        self.metric_key = metric_key

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.gt_boxes.append(y_true)
        self.pred_boxes.append(y_pred)

    def reset_state(self):
        self.gt_boxes = []
        self.pred_boxes = []

    def result(self):
        true_boxes = tf.concat([b["boxes"] for b in self.gt_boxes], axis=0)
        true_classes = tf.concat([b["classes"] for b in self.gt_boxes], axis=0)
        y_true = {"boxes": true_boxes, "classes": true_classes}

        pred_boxes = tf.concat([b["boxes"] for b in self.pred_boxes], axis=0)
        pred_classes = tf.concat([b["classes"] for b in self.pred_boxes], axis=0)
        pred_confidence = tf.concat([b["confidence"] for b in self.pred_boxes], axis=0)
        y_pred = {
            "boxes": pred_boxes,
            "classes": pred_classes,
            "confidence": pred_confidence,
        }
        with HidePrints():
            metric = compute_pycocotools_metric(y_true, y_pred, "xyxy", self.metric_key)
        return metric


def compute_pycocotools_metric(y_true, y_pred, bounding_box_format, key):
    box_pred = y_pred["boxes"]
    cls_pred = y_pred["classes"]
    confidence_pred = y_pred["confidence"]

    gt_boxes = y_true["boxes"]
    gt_classes = y_true["classes"]

    box_pred = bounding_box.convert_format(
        box_pred, source=bounding_box_format, target="yxyx"
    )
    gt_boxes = bounding_box.convert_format(
        gt_boxes, source=bounding_box_format, target="yxyx"
    )

    height = 640
    width = 640
    total_images = gt_boxes.shape[0]

    source_ids = tf.strings.as_string(
        tf.linspace(1, total_images, total_images), precision=0
    )

    ground_truth = {}
    ground_truth["source_id"] = [source_ids]
    ground_truth["height"] = [tf.tile(tf.constant([height]), [total_images])]
    ground_truth["width"] = [tf.tile(tf.constant([width]), [total_images])]

    ground_truth["num_detections"] = [
        tf.math.reduce_sum(tf.cast(y_true["classes"] != -1, tf.int32), axis=-1)
    ]
    ground_truth["boxes"] = [gt_boxes]
    ground_truth["classes"] = [gt_classes]

    predictions = {}
    predictions["source_id"] = [source_ids]
    predictions["detection_boxes"] = [box_pred]
    predictions["detection_classes"] = [cls_pred]
    predictions["detection_scores"] = [confidence_pred]
    predictions["num_detections"] = [
        tf.math.reduce_sum(tf.cast(y_pred["classes"] != -1, tf.int32), axis=-1)
    ]

    metrics = compute_pycoco_metrics(ground_truth, predictions)
    return metrics[key]
