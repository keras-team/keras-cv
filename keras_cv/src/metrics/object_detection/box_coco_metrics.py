# Copyright 2023 The KerasCV Authors
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

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from keras_cv.src import bounding_box
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import ops
from keras_cv.src.metrics import coco


class HidePrints:
    """A basic internal only context manager to hide print statements."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def _box_concat(boxes):
    """Concatenates two bounding box batches together."""
    result = {}
    for key in ["boxes", "classes"]:
        result[key] = tf.concat([b[key] for b in boxes], axis=0)

    if len(boxes) != 0 and "confidence" in boxes[0]:
        result["confidence"] = tf.concat(
            [b["confidence"] for b in boxes], axis=0
        )
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

METRIC_MAPPING = {
    "AP": "MaP",
    "AP50": "MaP@[IoU=50]",
    "AP75": "MaP@[IoU=75]",
    "APs": "MaP@[area=small]",
    "APm": "MaP@[area=medium]",
    "APl": "MaP@[area=large]",
    "ARmax1": "Recall@[max_detections=1]",
    "ARmax10": "Recall@[max_detections=10]",
    "ARmax100": "Recall@[max_detections=100]",
    "ARs": "Recall@[area=small]",
    "ARm": "Recall@[area=medium]",
    "ARl": "Recall@[area=large]",
}


@keras_cv_export("keras_cv.metrics.BoxCOCOMetrics")
class BoxCOCOMetrics(keras.metrics.Metric):
    """BoxCOCOMetrics computes standard object detection metrics.

    Args:
        bounding_box_format: the bounding box format for inputs.
        evaluate_freq: the number of steps to run before each evaluation.
            Due to the high computational cost of metric evaluation the final
            results are only updated once every `evaluate_freq` steps. Higher
            values will allow for faster training times, while lower numbers
            allow for higher numerical precision in metric reporting.

    Example:
    `BoxCOCOMetrics()` can be used like any standard metric with any
    KerasCV object detection model. Inputs to `y_true` must be KerasCV bounding
    box dictionaries, `{"classes": classes, "boxes": boxes}`, and `y_pred` must
    follow the same format with an additional `confidence` key.

    Unfortunately, at the moment `BoxCOCOMetrics()` are not TPU compatible with
    the `fit()` API. If you wish to evaluate `BoxCOCOMetrics()` for a model
    trained on TPU, we recommend using the `model.predict()` API and manually
    updating the metric state with the results.

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
        metrics=[keras_cv.metrics.BoxCOCOMetrics('xywh')]
    )
    model.fit(images, labels)
    ```
    """

    def __init__(self, bounding_box_format, evaluate_freq, name=None, **kwargs):
        if "dtype" not in kwargs:
            kwargs["dtype"] = "float32"
        super().__init__(name=name, **kwargs)
        self.ground_truths = []
        self.predictions = []
        self.bounding_box_format = bounding_box_format
        self.evaluate_freq = evaluate_freq
        self._eval_step_count = 0
        self._cached_result = [0] * len(METRIC_NAMES)

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
            y_pred_confidence = y_pred["confidence"]
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

        def result_on_host_cpu(force):
            with tf.device("/cpu:0"):
                # Without the call to `constant` `tf.py_function` selects the
                # first index automatically and just returns obj_result()[0]
                return tf.constant(obj_result(force), obj.dtype)

        obj.result_on_host_cpu = result_on_host_cpu

        def result_fn(self, force=False):
            py_func_result = tf.py_function(
                self.result_on_host_cpu, inp=[force], Tout=obj.dtype
            )
            result = {}
            for i, key in enumerate(METRIC_NAMES):
                result[self.name_prefix() + METRIC_MAPPING[key]] = (
                    py_func_result[i]
                )
            return result

        obj.result = types.MethodType(result_fn, obj)

        return obj

    def name_prefix(self):
        if self.name.startswith("box_coco_metrics"):
            return ""
        return self.name + "_"

    def update_state(self, y_true, y_pred, sample_weight=None):
        self._eval_step_count += 1

        if isinstance(y_true["boxes"], tf.RaggedTensor) != isinstance(
            y_pred["boxes"], tf.RaggedTensor
        ):
            # Make sure we have same ragged/dense status for y_true and y_pred
            y_true = bounding_box.to_dense(y_true)
            y_pred = bounding_box.to_dense(y_pred)

        self.ground_truths.append(y_true)
        self.predictions.append(y_pred)

        # Compute on first step, so we don't have an inconsistent list of
        # metrics in our train_step() results. This will just populate the
        # metrics with `0.0` until we get to `evaluate_freq`.
        if self._eval_step_count % self.evaluate_freq == 0:
            self._cached_result = self._compute_result()

    def reset_state(self):
        self.ground_truths = []
        self.predictions = []
        self._eval_step_count = 0
        self._cached_result = [0] * len(METRIC_NAMES)

    def result(self, force=False):
        if force:
            self._cached_result = self._compute_result()
        return self._cached_result

    def _compute_result(self):
        if len(self.predictions) == 0 or len(self.ground_truths) == 0:
            return dict([(key, 0) for key in METRIC_NAMES])
        with HidePrints():
            metrics = compute_pycocotools_metric(
                _box_concat(self.ground_truths),
                _box_concat(self.predictions),
                self.bounding_box_format,
            )
        results = []
        for key in METRIC_NAMES:
            # Workaround for the state where there are 0 boxes in a category.
            results.append(max(metrics[key], 0.0))
        return results


def compute_pycocotools_metric(y_true, y_pred, bounding_box_format):
    y_true = bounding_box.to_dense(y_true)
    y_pred = bounding_box.to_dense(y_pred)

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

    source_ids = np.char.mod("%d", np.linspace(1, total_images, total_images))

    ground_truth = {}
    ground_truth["source_id"] = [source_ids]

    ground_truth["num_detections"] = [
        ops.sum(ops.cast(y_true["classes"] >= 0, "int32"), axis=-1)
    ]
    ground_truth["boxes"] = [ops.convert_to_numpy(gt_boxes)]
    ground_truth["classes"] = [ops.convert_to_numpy(gt_classes)]

    predictions = {}
    predictions["source_id"] = [source_ids]
    predictions["detection_boxes"] = [ops.convert_to_numpy(box_pred)]
    predictions["detection_classes"] = [ops.convert_to_numpy(cls_pred)]
    predictions["detection_scores"] = [ops.convert_to_numpy(confidence_pred)]
    predictions["num_detections"] = [
        ops.sum(ops.cast(confidence_pred > 0, "int32"), axis=-1)
    ]

    return coco.compute_pycoco_metrics(ground_truth, predictions)
