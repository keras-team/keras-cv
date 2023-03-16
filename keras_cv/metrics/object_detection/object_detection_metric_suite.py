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
import os
import sys
import types

import tensorflow as tf
import tensorflow.keras as keras

import keras_cv
from keras_cv import bounding_box


class HidePrints:
    """A basic internal only context manager to hide print statements."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def _box_concat(b1, b2):
    """Concatenates two bounding box batches together."""
    if b1 is None:
        return b2
    if b2 is None:
        return b1

    result = {}
    for key in ["boxes", "classes", "confidence"]:
        result[key] = tf.concat([b1[key], b2[key]], axis=0)
    return result


METRIC_NAMES = [
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "ARmax1",
    "ARmax10",
    "ARmax100",
    "ARs",
    "ARm",
    "ARl",
]


class ObjectDetectionMetricSuite(keras.metrics.Metric):
    """ObjectDetectionMetricSuite computes standard object deteciton metrics.

    Args:
        bounding_box_format: the bounding box format for inputs.

    Usage:
    `ObjectDetectionMetricSuite()` can be used like any standard metric with any
    KerasCV object detection model.  Inputs to `y_true` must be KerasCV bounding
    box dictionaries, `{"classes": classes, "boxes": boxes}`, and `y_pred` must
    follow the same format with an additional `confidence` key.

    Using this metric suite alongside a model is trivial; simply provide it to
    the `compile()` arguments of the model:

    ```python
    images = tf.ones(shape=(1, 512, 512, 3))
    labels = {
        "boxes": [
            [
                [0, 0, 100, 100],
                [100, 100, 200, 200],
                [300, 300, 400, 400],
            ]
        ],
        "classes": [[1, 1, 1]],
    }
    model = keras_cv.models.RetinaNet(
        num_classes=20,
        bounding_box_format="xywh",
    )

    # Evaluate model
    model(images)

    # Train model
    model.compile(
        classification_loss='focal',
        box_loss='smoothl1',
        optimizer=tf.optimizers.SGD(global_clipnorm=10.0),
        jit_compile=False,
        metrics=[keras_cv.metrics.ObjectDetectionMetricSuite('xywh')]
    )
    model.fit(images, labels)
    ```
    """

    def __init__(self, bounding_box_format, **kwargs):
        super().__init__(**kwargs)
        self.ground_truths = None
        self.predictions = None
        self.bounding_box_format = bounding_box_format

    def __new__(cls, *args, **kwargs):
        obj = super(keras.metrics.Metric, cls).__new__(cls)

        # Wrap the update_state function in a py_function and scope it to /cpu:0
        obj_update_state = obj.update_state

        def update_state_on_cpu(
            y_true_boxes,
            y_true_classes,
            y_pred_boxes,
            y_pred_classes,
            y_pred_confidence,
            sample_weight=None,
        ):
            y_true = {"boxes": y_true_boxes, "classes": y_true_classes}
            y_pred = {
                "boxes": y_pred_boxes,
                "classes": y_pred_classes,
                "confidence": y_pred_confidence,
            }
            with tf.device("/cpu:0"):
                return obj_update_state(y_true, y_pred, sample_weight)

        obj.update_state_on_cpu = update_state_on_cpu

        def update_state_fn(self, y_true, y_pred, sample_weight=None):
            y_true_boxes = y_true["boxes"]
            y_true_classes = y_true["classes"]
            y_pred_boxes = y_pred["boxes"]
            y_pred_classes = y_pred["classes"]
            y_pred_confidence = y_pred["classes"]
            eager_inputs = [
                y_true_boxes,
                y_true_classes,
                y_pred_boxes,
                y_pred_classes,
                y_pred_confidence,
            ]
            if sample_weight is not None:
                eager_inputs.append(sample_weight)
            return tf.py_function(
                func=self.update_state_on_cpu, inp=eager_inputs, Tout=[]
            )

        obj.update_state = types.MethodType(update_state_fn, obj)

        # Wrap the result function in a py_function and scope it to /cpu:0
        obj_result = obj.result

        def result_on_host_cpu():
            with tf.device("/cpu:0"):
                # Without the call to `constant` `tf.py_function` selects the
                # first index automatically and just returns obj_result()[0]
                return tf.constant(obj_result(), obj.dtype)

        obj.result_on_host_cpu = result_on_host_cpu

        def result_fn(self):
            py_func_result = tf.py_function(
                self.result_on_host_cpu, inp=[], Tout=obj.dtype
            )
            result = {}
            for i, key in enumerate(METRIC_NAMES):
                result[key] = py_func_result[i]
            return result

        obj.result = types.MethodType(result_fn, obj)

        return obj

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.ground_truths = _box_concat(self.ground_truths, y_true)
        self.predictions = _box_concat(self.predictions, y_pred)

    def reset_state(self):
        self.ground_truths = None
        self.predictions = None

    def result(self):
        if self.predictions is None or self.ground_truths is None:
            return dict([(key, 0) for key in METRIC_NAMES])
        with HidePrints():
            metrics = compute_pycocotools_metric(
                self.ground_truths, self.predictions, self.bounding_box_format
            )
        results = []
        for key in METRIC_NAMES:
            results.append(metrics[key])
        return results


def compute_pycocotools_metric(y_true, y_pred, bounding_box_format):
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

    total_images = gt_boxes.shape[0]

    source_ids = tf.strings.as_string(
        tf.linspace(1, total_images, total_images), precision=0
    )

    ground_truth = {}
    ground_truth["source_id"] = [source_ids]

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

    return keras_cv.metrics.coco.compute_pycoco_metrics(
        ground_truth, predictions
    )
