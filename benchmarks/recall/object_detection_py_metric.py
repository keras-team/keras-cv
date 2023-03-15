# Copyright 2023 The Keras Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base class for Python-based metrics"""

import types

import tensorflow.compat.v2 as tf
from keras.metrics import base_metric
from tensorflow.python.util.tf_export import keras_export


class ODPyMetric(base_metric.Metric):
    """`ODPyMetric` is an internal facing fork of `PyMetric`.

    `ODPyMetric` is implemented to support the fact that `tf.py_function` cannot
    accept dictionaries as an input.
    """

    def __init__(self, name=None, dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.reset_state()

    def __new__(cls, *args, **kwargs):
        obj = super(base_metric.Metric, cls).__new__(cls)

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
                return obj_result()

        obj.result_on_host_cpu = result_on_host_cpu

        def result_fn(self):
            return tf.py_function(self.result_on_host_cpu, inp=[], Tout=obj.dtype)

        obj.result = types.MethodType(result_fn, obj)

        return obj

    def update_state(self, y_true, y_pred, sample_weight=None):
        raise NotImplementedError("Subclasses should implement `update_state`")

    def merge_state(self, metrics):
        raise NotImplementedError("Subclasses should implement `merge_state`")

    def reset_state(self):
        raise NotImplementedError("Subclasses should implement `reset_state`")

    def result(self):
        raise NotImplementedError("Subclasses should implement `result`")
